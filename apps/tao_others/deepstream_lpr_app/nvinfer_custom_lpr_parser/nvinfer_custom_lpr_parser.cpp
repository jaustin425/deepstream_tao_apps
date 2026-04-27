/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#include <assert.h>
#include <codecvt>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <locale>
#include <cstdio>
#include <string>
#include <cstring>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include "nvdsinfer.h"

static bool dict_ready = false;
std::vector<std::string> dict_table;

static bool readable_file(const std::string &path) {
  return !path.empty() && access(path.c_str(), R_OK) == 0;
}

static std::string parent_dir(const std::string &path) {
  const std::string::size_type slash = path.find_last_of('/');
  return slash == std::string::npos ? std::string(".") : path.substr(0, slash);
}

static std::string join_path(const std::string &base, const std::string &leaf) {
  if (base.empty() || base == ".") {
    return leaf;
  }
  if (base.back() == '/') {
    return base + leaf;
  }
  return base + "/" + leaf;
}

static std::string resolve_dict_path(std::vector<std::string> &searched_paths) {
  const char *override_path = getenv("NVDSINFER_LPR_DICT_PATH");
  if (override_path != NULL && override_path[0] != '\0') {
    searched_paths.emplace_back(override_path);
    if (readable_file(override_path)) {
      return override_path;
    }
  }

  Dl_info dl_info;
  if (dladdr(reinterpret_cast<void *>(resolve_dict_path), &dl_info) != 0 &&
      dl_info.dli_fname != NULL) {
    const std::string lib_dir = parent_dir(dl_info.dli_fname);
    searched_paths.push_back(join_path(lib_dir, "../models/LP/LPR/labels_us.txt"));
    searched_paths.push_back(join_path(lib_dir, "../dict.txt"));
  }

  searched_paths.push_back("models/LP/LPR/labels_us.txt");
  searched_paths.push_back("labels_us.txt");
  searched_paths.push_back("dict.txt");

  for (const std::string &candidate : searched_paths) {
    if (readable_file(candidate)) {
      return candidate;
    }
  }

  return "";
}

extern "C" bool NvDsInferParseCustomNVPlate(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo, float classifierThreshold,
    std::vector<NvDsInferAttribute> &attrList, std::string &attrString) {
  const int *outputStrBuffer = NULL;
  const int64_t *outputStrBuffer64 = NULL;
  const float *outputConfBuffer = NULL;
  NvDsInferAttribute LPR_attr;

  int seq_len = 0;
  int str_seq_len = 0;
  int conf_seq_len = 0;

  // Get list
  std::vector<int> str_idxes;
  int prev = 100;

  // For confidence
  std::vector<double> bank_softmax_max;
  bool do_softmax = false;
  std::ifstream fdict;

  setlocale(LC_CTYPE, "");

  if (!dict_ready) {
    std::vector<std::string> searched_paths;
    const std::string dict_path = resolve_dict_path(searched_paths);
    if (!dict_path.empty()) {
      fdict.open(dict_path.c_str());
    }
    if (!fdict.is_open()) {
      std::string message = "open dictionary file failed. searched:";
      for (const std::string &candidate : searched_paths) {
        message += " ";
        message += candidate;
      }
      message += "\n";
      fputs(message.c_str(), stdout);
      return false;
    }
    while (!fdict.eof()) {
      std::string strLineAnsi;
      if (getline(fdict, strLineAnsi)) {
        dict_table.push_back(strLineAnsi);
      }
    }
    if (dict_table.empty()) {
      printf("dictionary file is empty. %s\n", dict_path.c_str());
      fdict.close();
      return false;
    }
    dict_ready = true;
    fdict.close();
  }

  int layer_size = outputLayersInfo.size();

  LPR_attr.attributeConfidence = 1.0;

  seq_len = networkInfo.width / 4;

  for (int li = 0; li < layer_size; li++) {
    if (!outputLayersInfo[li].isInput) {
      if (outputLayersInfo[li].dataType == 0) {
        if (!outputConfBuffer) {
          outputConfBuffer = static_cast<float *>(outputLayersInfo[li].buffer);
          conf_seq_len = static_cast<int>(outputLayersInfo[li].inferDims.numElements);
        }
      } else if (outputLayersInfo[li].dataType == 3) {
        if (!outputStrBuffer) {
          outputStrBuffer = static_cast<int *>(outputLayersInfo[li].buffer);
          str_seq_len = static_cast<int>(outputLayersInfo[li].inferDims.numElements);
        }
      } else if (outputLayersInfo[li].dataType == 4) {
        if (!outputStrBuffer64) {
          outputStrBuffer64 = static_cast<int64_t *>(outputLayersInfo[li].buffer);
          str_seq_len = static_cast<int>(outputLayersInfo[li].inferDims.numElements);
        }
      }
    }
  }

  if ((!outputStrBuffer && !outputStrBuffer64) || !outputConfBuffer) {
    return false;
  }

  if (str_seq_len > 0 && conf_seq_len > 0) {
    seq_len = std::min(str_seq_len, conf_seq_len);
  } else if (str_seq_len > 0) {
    seq_len = str_seq_len;
  } else if (conf_seq_len > 0) {
    seq_len = conf_seq_len;
  } else {
    seq_len = static_cast<int>(networkInfo.width / 4);
  }

  if (seq_len <= 0) {
    return false;
  }

  bank_softmax_max.reserve(seq_len);

  for (int seq_id = 0; seq_id < seq_len; seq_id++) {
    do_softmax = false;

    int64_t curr_data_raw = outputStrBuffer64
                                ? outputStrBuffer64[seq_id]
                                : static_cast<int64_t>(outputStrBuffer[seq_id]);
    if (curr_data_raw < 0 ||
        curr_data_raw > static_cast<int64_t>(dict_table.size())) {
      continue;
    }
    int curr_data = static_cast<int>(curr_data_raw);
    if (seq_id == 0) {
      prev = curr_data;
      str_idxes.push_back(curr_data);
      if (curr_data != static_cast<int>(dict_table.size()))
        do_softmax = true;
    } else {
      if (curr_data != prev) {
        str_idxes.push_back(curr_data);
        if (static_cast<unsigned long>(curr_data) != dict_table.size())
          do_softmax = true;
      }
      prev = curr_data;
    }

    // Do softmax
    if (do_softmax) {
      do_softmax = false;
      bank_softmax_max.push_back(outputConfBuffer[seq_id]);
    }
  }

  attrString = "";
  for (unsigned int id = 0; id < str_idxes.size(); id++) {
    if (static_cast<unsigned int>(str_idxes[id]) != dict_table.size()) {
      attrString += dict_table[str_idxes[id]];
    }
  }

  // Ignore the short string, it may be wrong plate string
  if (bank_softmax_max.size() >= 3) {
    LPR_attr.attributeIndex = 0;
    LPR_attr.attributeValue = 1;
    LPR_attr.attributeLabel = strdup(attrString.c_str());
    for (double confidence : bank_softmax_max) {
      LPR_attr.attributeConfidence *= confidence;
    }
    attrList.push_back(LPR_attr);
  }

  return true;
}
