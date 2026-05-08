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
#include <glib.h>
#include <glib-unix.h>
#include <gmodule.h>
#include <gst/gst.h>
#include <locale.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime_api.h>

#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <cctype>
#include <iomanip>
#include <limits>
#include <regex>
#include <algorithm>
#include <vector>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "ds_yml_parse.h"
#include "evidence_logger.hpp"
#include "gst-nvmessage.h"
#include "gstnvdsmeta.h"
#include "nvbufsurface.h"
#include "nvds_yml_parser.h"
#include "nvdsmeta.h"

#define MAX_DISPLAY_LEN 64

#define MEASURE_ENABLE 1

#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_PERSON 2

#define SGIE_CLASS_ID_LPD 0
#define PRIMARY_DETECTOR_UID 1
#define SECONDARY_DETECTOR_UID 2
#define SECONDARY_CLASSIFIER_UID 3
#define VEHICLE_COLOR_CLASSIFIER_UID 4
#define VEHICLE_TYPE_CLASSIFIER_UID 5
#define VEHICLE_MAKE_CLASSIFIER_UID 6

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. Match the primary detector input size so the live
 * path does not require an extra GPU resize inside nvinfer. */
#define MUXER_OUTPUT_WIDTH 960
#define MUXER_OUTPUT_HEIGHT 544

/* Match the Arducam USB camera mode that is already proven stable through the
 * direct-preview path on this host to avoid renegotiation and high-bandwidth
 * live ingest instability. */
#define CAMERA_SOURCE_WIDTH 960
#define CAMERA_SOURCE_HEIGHT 600

/* Muxer batch formation timeout. Keep this near one 30 FPS frame interval so a
 * live patrol read does not wait behind stale frames while forming batches. */
#define MUXER_BATCH_TIMEOUT_USEC 40000

/* Check for parsing error. */
#define RETURN_ON_PARSER_ERROR(parse_expr)                                     \
  if (NVDS_YAML_PARSER_SUCCESS != parse_expr) {                                \
    g_printerr("Error in parsing configuration file.\n");                      \
    return -1;                                                                 \
  }

gint frame_number = 0;
gint total_plate_number = 0;
gchar pgie_classes_str[4][32] = {"Vehicle", "TwoWheeler", "Person", "Roadsign"};

struct PlateTrack {
    std::map<std::string, int> votes;

    std::map<int, std::map<char, int>> char_votes;
  std::map<std::string, int> make_votes;
  std::map<std::string, int> type_votes;
  std::map<std::string, int> color_votes;
  int large_vehicle_hint_score = 0;
  std::string hinted_large_vehicle_type;
  float hinted_large_vehicle_type_probability = 0.0f;
  std::string hinted_large_vehicle_make;
  float hinted_large_vehicle_make_probability = 0.0f;

    bool reported = false;
  bool debug_reported = false;
    int last_seen_frame = 0;
    int first_seen_frame = -1;
    gint64 first_seen_time_us = 0;
    int first_readable_frame = -1;
    gint64 first_readable_time_us = 0;
    int first_confirmed_frame = -1;
    gint64 first_confirmed_time_us = 0;
    int stable_frames = 0;
    std::string last_best;

  bool locked = false;
  std::string locked_plate;
  int locked_confidence = 0;
  int lock_frame = -1;
  int missed_frames = 0;
  std::string last_debug_plate;
  int last_debug_confidence = 0;
  int last_debug_frame = -1;
  std::string last_confirmed_plate;
  int last_confirmed_confidence = 0;
  int last_confirmed_frame = -1;
  std::string last_locked_plate;
  int last_locked_confidence = 0;
  int last_locked_frame = -1;
};

struct ConfirmedPlate {
  std::string plate;
  int confidence = 0;
  int votes = 0;
};

struct PlateConfidenceResult {
    int score = 0;
    int vote_points = 0;
    int stability_points = 0;
    int dominance_points = 0;
    int separation_points = 0;
  int bonus_points = 0;
    int penalty_points = 0;
    int ca_pattern_points = 0;
    std::string ca_pattern_name;  // best matching CA pattern label
};

struct VehicleAttributes {
  std::string make;
  std::string type;
  std::string color;
};

struct RecentPlateAttributes {
  VehicleAttributes attributes;
  std::string video_source;
  int frame_number = 0;
  int confidence = 0;
};

struct RecentPlateEvent {
  std::string video_source;
  std::string plate;
  std::string event_type;
  int confidence = 0;
  int first_frame = 0;
  int last_seen_frame = 0;
  int last_emit_frame = 0;
};

struct ClassifierLabelResult {
  std::string label;
  float probability = 0.0f;
};

struct VehicleAttributeObservations {
  ClassifierLabelResult make;
  ClassifierLabelResult type;
  ClassifierLabelResult color;
};

struct StabilizedAttributeResult {
  std::string label;
  int best_votes = 0;
  int second_votes = 0;
  int total_votes = 0;
};

struct VehicleAttributeVoteContext {
  bool allow_make = false;
  bool allow_type = false;
  bool allow_color = false;
  bool relaxed_large_vehicle_make = false;
  int make_bonus = 0;
  int type_bonus = 0;
  int color_bonus = 0;
};

struct VehicleCropQuality {
  bool valid = false;
  bool likely_infrared = false;
  bool clipped = false;
  bool heavy_glare = false;
  bool heavy_shadow = false;
  bool low_contrast = false;
  float score = 0.0f;
};

struct AlprRuntimeConfig {
  std::string profile = "patrol-fast";
  bool low_latency_rtsp = true;
  int rtsp_latency_ms = 80;
  bool rtsp_drop_on_latency = true;
  int rtsp_protocols = 4;  // GST_RTSP_LOWER_TRANS_TCP
  bool latency_metrics = true;
};

static std::map<std::string, PlateTrack> plate_tracks;
static std::map<std::string, RecentPlateAttributes> recent_plate_attributes;
static std::vector<RecentPlateEvent> recent_plate_events;
static std::string g_evidence_root = "evidence";
static AlprRuntimeConfig g_runtime_config;
static const char *kModelVersion = "alpr_ds_v1";

static const int REJECT_THRESHOLD = 60;
static const int DEBUG_THRESHOLD_MIN = 60;
static const int DEBUG_STABLE_FRAMES = 4;
static const int LOCK_THRESHOLD = 82;
static const int LOCK_STABLE_FRAMES = 8;
static const int LOCK_HOLD_FRAMES = 20;
static const int TRACK_DECAY_FRAMES = 15;
static const int TRACK_DROP_FRAMES = 40;
static const int TRACK_EVENT_COOLDOWN_FRAMES = 120;
static const int TRACK_EVENT_IMPROVEMENT_DELTA = 2;
static const int TRACK_LOCK_IMPROVEMENT_DELTA = 8;
static const int RECENT_PLATE_ATTRIBUTE_MAX_AGE_FRAMES = 1800;
static const int SOURCE_PLATE_EVENT_SUPPRESS_FRAMES = 180;
static const int SOURCE_PLATE_EVENT_MIN_REEMIT_FRAMES = 180;
static const int SOURCE_PLATE_EVENT_IMPROVEMENT_DELTA = 12;

static std::string preview_focus_state_for_plate(int confidence, bool readable) {
  if (confidence >= REJECT_THRESHOLD) {
    return "in_focus";
  }
  if (confidence >= DEBUG_THRESHOLD_MIN || readable) {
    return "approaching_focus";
  }
  return "out_of_focus";
}

static std::string get_plate_string(NvDsLabelInfo *label_info) {
    if (!label_info || !label_info->result_label)
        return std::string();
    return std::string(label_info->result_label);
}

static cv::Mat extract_frame_mat(GstBuffer *buf, guint batch_id) {
  GstMapInfo in_map_info = GST_MAP_INFO_INIT;
  if (!gst_buffer_map(buf, &in_map_info, GST_MAP_READ)) {
    return cv::Mat();
  }

  NvBufSurface *surface = reinterpret_cast<NvBufSurface *>(in_map_info.data);
  if (!surface || batch_id >= surface->batchSize) {
    gst_buffer_unmap(buf, &in_map_info);
    return cv::Mat();
  }

  if (NvBufSurfaceMap(surface, batch_id, -1, NVBUF_MAP_READ) != 0) {
    gst_buffer_unmap(buf, &in_map_info);
    return cv::Mat();
  }

  if ((surface->memType == NVBUF_MEM_SURFACE_ARRAY ||
       surface->memType == NVBUF_MEM_HANDLE) &&
      NvBufSurfaceSyncForCpu(surface, batch_id, 0) != 0) {
    NvBufSurfaceUnMap(surface, batch_id, -1);
    gst_buffer_unmap(buf, &in_map_info);
    return cv::Mat();
  }

  NvBufSurfaceParams &params = surface->surfaceList[batch_id];
  void *mapped_ptr = params.mappedAddr.addr[0];
  if (!mapped_ptr) {
    NvBufSurfaceUnMap(surface, batch_id, -1);
    gst_buffer_unmap(buf, &in_map_info);
    return cv::Mat();
  }

  /* Expect RGBA here from the pre-OSD convert branch. */
  cv::Mat frame_rgba(params.height, params.width, CV_8UC4, mapped_ptr, params.pitch);
  cv::Mat frame_bgr;
  cv::cvtColor(frame_rgba, frame_bgr, cv::COLOR_RGBA2BGR);
  cv::Mat frame_copy = frame_bgr.clone();

  NvBufSurfaceUnMap(surface, batch_id, -1);
  gst_buffer_unmap(buf, &in_map_info);
  return frame_copy;
}

static std::string build_video_source_name(const NvDsFrameMeta *frame_meta) {
  return std::string("source_") + std::to_string(frame_meta ? frame_meta->source_id : 0);
}

static std::string build_plate_track_key(const std::string& video_source,
                                         bool track_id_valid,
                                         uint64_t track_id,
                                         const NvDsObjectMeta* plate_meta) {
  if (track_id_valid) {
    return video_source + ":track:" + std::to_string(track_id);
  }

  std::ostringstream out;
  out << video_source << ":bbox:"
      << static_cast<int>(plate_meta ? plate_meta->rect_params.left / 32.0f : 0) << ':'
      << static_cast<int>(plate_meta ? plate_meta->rect_params.top / 32.0f : 0) << ':'
      << static_cast<int>(plate_meta ? plate_meta->rect_params.width / 16.0f : 0) << ':'
      << static_cast<int>(plate_meta ? plate_meta->rect_params.height / 16.0f : 0);
  return out.str();
}

static std::string trim_copy(const std::string& value) {
  std::string::size_type begin = 0;
  while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin]))) {
    ++begin;
  }

  std::string::size_type end = value.size();
  while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
    --end;
  }

  return value.substr(begin, end - begin);
}

static std::string to_lower_copy(const std::string& value) {
  std::string lowered = value;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                 [](unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });
  return lowered;
}

static std::string title_case_words(const std::string& value) {
  std::string label = value;
  bool previous_was_space = true;
  for (char& ch : label) {
    if (std::isspace(static_cast<unsigned char>(ch))) {
      previous_was_space = true;
      continue;
    }

    ch = previous_was_space
             ? static_cast<char>(std::toupper(static_cast<unsigned char>(ch)))
             : static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    previous_was_space = false;
  }
  return label;
}

static float clamp_probability(float value) {
  if (!std::isfinite(value) || value <= 0.0f) {
    return 0.0f;
  }
  return std::max(0.0f, std::min(value, 1.0f));
}

static bool is_valid_plate_text(const std::string& text);
static cv::Rect clamp_rect_to_frame(const cv::Mat& frame,
                                    int left,
                                    int top,
                                    int width,
                                    int height);
static void decay_attribute_votes(std::map<std::string, int>& vote_map);
static std::string canonical_vehicle_make_for_large_vehicle(const std::string& label);
static bool is_plausible_large_vehicle_make(const std::string& label);
static std::string normalize_plate(std::string p);

static std::string normalize_vehicle_attribute_label(const std::string& raw_label,
                                                     guint classifier_uid) {
  std::string label = trim_copy(raw_label);
  if (label.empty()) {
    return label;
  }

  std::replace(label.begin(), label.end(), '_', ' ');
  std::replace(label.begin(), label.end(), '-', ' ');
  label = trim_copy(label);

  std::string normalized_key = to_lower_copy(label);

  if (classifier_uid == VEHICLE_MAKE_CLASSIFIER_UID) {
    if (normalized_key == "gmc") {
      return "GMC";
    }
    if (normalized_key == "ram" || normalized_key == "ram trucks" ||
        normalized_key == "dodge ram") {
      return "RAM";
    }
    if (normalized_key == "bmw") {
      return "BMW";
    }
    if (normalized_key == "chevy") {
      return "Chevrolet";
    }
    if (normalized_key == "mercedes") {
      return "Mercedes-Benz";
    }
  }

  if (classifier_uid == VEHICLE_TYPE_CLASSIFIER_UID) {
    if (normalized_key == "suv") {
      return "SUV";
    }
    if (normalized_key == "largevehicle") {
      return "Large Vehicle";
    }
  }

  if (classifier_uid == VEHICLE_COLOR_CLASSIFIER_UID) {
    if (normalized_key == "grey") {
      return "Gray";
    }
  }

  return title_case_words(label);
}

static bool config_has_section(const std::string& config_path, const char* section_name) {
  try {
    YAML::Node root = YAML::LoadFile(config_path);
    YAML::Node section = root[section_name];
    return section && !section.IsNull();
  } catch (const std::exception&) {
    return false;
  }
}

static bool yaml_bool_or_default(const YAML::Node& node, bool fallback) {
  if (!node || node.IsNull()) {
    return fallback;
  }

  try {
    return node.as<bool>();
  } catch (const std::exception&) {
    try {
      std::string value = to_lower_copy(trim_copy(node.as<std::string>()));
      if (value == "1" || value == "true" || value == "yes" || value == "on") {
        return true;
      }
      if (value == "0" || value == "false" || value == "no" || value == "off") {
        return false;
      }
    } catch (const std::exception&) {
    }
  }

  return fallback;
}

static int yaml_int_or_default(const YAML::Node& node, int fallback) {
  if (!node || node.IsNull()) {
    return fallback;
  }

  try {
    return node.as<int>();
  } catch (const std::exception&) {
    return fallback;
  }
}

static int parse_rtsp_protocols(const std::string& value, int fallback) {
  std::string normalized = to_lower_copy(trim_copy(value));
  if (normalized == "tcp") {
    return 4;
  }
  if (normalized == "udp") {
    return 1;
  }
  if (normalized == "udp-mcast" || normalized == "multicast") {
    return 2;
  }
  if (normalized == "auto" || normalized == "any") {
    return 0x7;
  }
  return fallback;
}

static AlprRuntimeConfig load_alpr_runtime_config(const std::string& config_path) {
  AlprRuntimeConfig config;

  try {
    YAML::Node root = YAML::LoadFile(config_path);
    YAML::Node runtime = root["alpr-runtime"];
    if (!runtime || !runtime.IsMap()) {
      runtime = root["runtime"];
    }
    if (!runtime || !runtime.IsMap()) {
      return config;
    }

    if (runtime["profile"] && runtime["profile"].IsScalar()) {
      config.profile = runtime["profile"].as<std::string>();
    }

    std::string normalized_profile = to_lower_copy(trim_copy(config.profile));
    if (normalized_profile == "balanced" || normalized_profile == "review") {
      config.low_latency_rtsp = false;
      config.rtsp_latency_ms = 200;
      config.rtsp_drop_on_latency = false;
      config.rtsp_protocols = 4;
    } else {
      config.profile = config.profile.empty() ? "patrol-fast" : config.profile;
      config.low_latency_rtsp = true;
      config.rtsp_latency_ms = 80;
      config.rtsp_drop_on_latency = true;
      config.rtsp_protocols = 4;
      config.latency_metrics = true;
    }

    config.low_latency_rtsp =
      yaml_bool_or_default(runtime["low-latency-rtsp"], config.low_latency_rtsp);
    config.rtsp_latency_ms =
      yaml_int_or_default(runtime["rtsp-latency-ms"], config.rtsp_latency_ms);
    config.rtsp_drop_on_latency =
      yaml_bool_or_default(runtime["rtsp-drop-on-latency"],
                           config.rtsp_drop_on_latency);
    if (runtime["rtsp-transport"] && runtime["rtsp-transport"].IsScalar()) {
      config.rtsp_protocols =
        parse_rtsp_protocols(runtime["rtsp-transport"].as<std::string>(),
                             config.rtsp_protocols);
    }
    config.latency_metrics =
      yaml_bool_or_default(runtime["latency-metrics"], config.latency_metrics);
  } catch (const std::exception& e) {
    g_printerr("Failed to load alpr-runtime config from %s: %s\n",
               config_path.c_str(), e.what());
  }

  if (config.rtsp_latency_ms < 0) {
    config.rtsp_latency_ms = 0;
  }

  return config;
}

static bool link_element_chain(const std::vector<GstElement*>& elements) {
  if (elements.size() < 2) {
    return true;
  }

  for (size_t i = 0; i + 1 < elements.size(); ++i) {
    if (!elements[i] || !elements[i + 1] || !gst_element_link(elements[i], elements[i + 1])) {
      return false;
    }
  }

  return true;
}

static bool is_v4l2_camera_source(const std::string& source_path) {
  return source_path.find("/dev/video") != std::string::npos ||
         source_path.find("/dev/v4l/") != std::string::npos;
}

static std::string normalize_v4l2_camera_source(const std::string& source_path) {
  std::string device_path = source_path;

  std::string::size_type v4l2_prefix_pos = device_path.find("v4l2://");
  if (v4l2_prefix_pos != std::string::npos) {
    device_path = device_path.substr(v4l2_prefix_pos + 7);
  }

  if (device_path.rfind("/dev/v4l/", 0) == 0) {
    char *resolved_path = realpath(device_path.c_str(), NULL);
    if (resolved_path != NULL) {
      std::string canonical_path(resolved_path);
      free(resolved_path);
      return canonical_path;
    }
    return device_path;
  }

  std::string::size_type device_pos = device_path.find("/dev/video");
  if (device_pos != std::string::npos) {
    return device_path.substr(device_pos);
  }

  return device_path;
}

static bool is_uri_source(const std::string& source_path) {
  return gst_uri_is_valid(source_path.c_str());
}

static bool is_live_uri_source(const std::string& source_path) {
  return g_str_has_prefix(source_path.c_str(), "rtsp://") ||
         g_str_has_prefix(source_path.c_str(), "rtsps://") ||
         g_str_has_prefix(source_path.c_str(), "udp://");
}

static bool is_rtsp_uri_source(const std::string& source_path) {
  return g_str_has_prefix(source_path.c_str(), "rtsp://") ||
         g_str_has_prefix(source_path.c_str(), "rtsps://");
}

static void apply_live_dashboard_config(const std::string& config_path) {
  try {
    YAML::Node root = YAML::LoadFile(config_path);
    YAML::Node live_dashboard = root["live-dashboard"];
    if (!live_dashboard || !live_dashboard.IsMap()) {
      return;
    }

    YAML::Node endpoint = live_dashboard["endpoint"];
    if (endpoint && endpoint.IsScalar()) {
      std::string endpoint_value = endpoint.as<std::string>();
      if (!endpoint_value.empty()) {
        setenv("ALPR_LIVE_ENDPOINT", endpoint_value.c_str(), 1);
      }
    }

    YAML::Node source_map = live_dashboard["source-map"];
    if (source_map && source_map.IsMap()) {
      std::ostringstream mapping;
      bool first = true;
      for (YAML::const_iterator it = source_map.begin(); it != source_map.end(); ++it) {
        if (!it->first.IsScalar() || !it->second.IsScalar()) {
          continue;
        }
        if (!first) {
          mapping << ',';
        }
        first = false;
        mapping << it->first.as<std::string>() << '=' << it->second.as<std::string>();
      }
      std::string mapping_value = mapping.str();
      if (!mapping_value.empty()) {
        setenv("ALPR_LIVE_SOURCE_MAP", mapping_value.c_str(), 1);
      }
    }
  } catch (const std::exception& exc) {
    g_printerr("Warning: failed to parse live-dashboard config from %s: %s\n",
               config_path.c_str(), exc.what());
  }
}

static ClassifierLabelResult get_classifier_result(NvDsObjectMeta* obj_meta,
                                                   guint classifier_uid) {
  ClassifierLabelResult result;
  if (!obj_meta) {
    return result;
  }

  for (NvDsMetaList* classifier_list = obj_meta->classifier_meta_list;
       classifier_list != NULL;
       classifier_list = classifier_list->next) {
    NvDsClassifierMeta* classifier_meta =
        static_cast<NvDsClassifierMeta*>(classifier_list->data);
    if (!classifier_meta ||
      static_cast<guint>(classifier_meta->unique_component_id) != classifier_uid) {
      continue;
    }

    for (NvDsMetaList* label_list = classifier_meta->label_info_list;
         label_list != NULL;
         label_list = label_list->next) {
      NvDsLabelInfo* label_info = static_cast<NvDsLabelInfo*>(label_list->data);
      if (!label_info || !label_info->result_label) {
        continue;
      }

      std::string label = normalize_vehicle_attribute_label(label_info->result_label,
                                                            classifier_uid);
      if (!label.empty()) {
        result.label = label;
        result.probability = clamp_probability(label_info->result_prob);
        return result;
      }
    }
  }

  return result;
}

static VehicleAttributeObservations extract_vehicle_attribute_observations(
    NvDsObjectMeta* vehicle_meta) {
  VehicleAttributeObservations observations;
  if (!vehicle_meta) {
    return observations;
  }

  observations.color = get_classifier_result(vehicle_meta, VEHICLE_COLOR_CLASSIFIER_UID);
  observations.type = get_classifier_result(vehicle_meta, VEHICLE_TYPE_CLASSIFIER_UID);
  observations.make = get_classifier_result(vehicle_meta, VEHICLE_MAKE_CLASSIFIER_UID);
  return observations;
}

static float minimum_attribute_probability(guint classifier_uid) {
  switch (classifier_uid) {
    case VEHICLE_COLOR_CLASSIFIER_UID:
      return 0.60f;
    case VEHICLE_TYPE_CLASSIFIER_UID:
      return 0.58f;
    case VEHICLE_MAKE_CLASSIFIER_UID:
      return 0.72f;
    default:
      return 0.0f;
  }
}

static int attribute_vote_weight(guint classifier_uid, float probability) {
  float clamped_probability = clamp_probability(probability);
  int weight = 1;

  if (clamped_probability > 0.0f) {
    weight = std::max(1, static_cast<int>((clamped_probability * 10.0f) + 0.5f));
  }

  if (classifier_uid == VEHICLE_COLOR_CLASSIFIER_UID && clamped_probability >= 0.7f) {
    weight += 1;
  }

  if (classifier_uid == VEHICLE_MAKE_CLASSIFIER_UID && clamped_probability >= 0.8f) {
    weight += 1;
  }

  return weight;
}

static int get_vote_count_for_label(const std::map<std::string, int>& vote_map,
                                    const std::string& label) {
  std::map<std::string, int>::const_iterator it = vote_map.find(label);
  if (it == vote_map.end()) {
    return 0;
  }
  return it->second;
}

static bool env_flag_enabled(const char* name) {
  const char* value = std::getenv(name);
  if (!value) {
    return false;
  }

  std::string lowered = to_lower_copy(trim_copy(value));
  return !lowered.empty() && lowered != "0" && lowered != "false" &&
         lowered != "off" && lowered != "no";
}

static bool should_log_large_vehicle_make_debug(const std::string& plate) {
  if (!env_flag_enabled("ALPR_DEBUG_LARGE_VEHICLE_MAKE")) {
    return false;
  }

  const char* filter = std::getenv("ALPR_DEBUG_LARGE_VEHICLE_MAKE_PLATE");
  if (!filter || !*filter) {
    return true;
  }

  return normalize_plate(plate) == normalize_plate(std::string(filter));
}

static bool verbose_frame_logs_enabled() {
  static const bool enabled = env_flag_enabled("ALPR_VERBOSE_FRAME_LOGS");
  return enabled;
}

static bool debug_events_enabled() {
  static const bool enabled = env_flag_enabled("ALPR_WRITE_DEBUG_EVENTS");
  return enabled;
}

static bool latency_metrics_enabled() {
  return g_runtime_config.latency_metrics ||
         env_flag_enabled("ALPR_LATENCY_METRICS");
}

static void log_plate_latency_event(const char* event_type,
                                    const std::string& video_source,
                                    const std::string& plate,
                                    const PlateTrack& track,
                                    int current_frame,
                                    gint64 current_time_us,
                                    int confidence,
                                    int best_votes,
                                    int stable_frames) {
  if (!latency_metrics_enabled()) {
    return;
  }

  int frames_from_first_read = -1;
  gint64 ms_from_first_read = -1;
  if (track.first_seen_frame >= 0 && track.first_seen_time_us > 0) {
    frames_from_first_read = current_frame - track.first_seen_frame;
    ms_from_first_read = (current_time_us - track.first_seen_time_us) / 1000;
  }

  int frames_from_readable = -1;
  gint64 ms_from_readable = -1;
  if (track.first_readable_frame >= 0 && track.first_readable_time_us > 0) {
    frames_from_readable = current_frame - track.first_readable_frame;
    ms_from_readable = (current_time_us - track.first_readable_time_us) / 1000;
  }

  std::cout << "ALPR_LATENCY event=" << event_type
            << " source=" << video_source
            << " plate=" << plate
            << " confidence=" << confidence
            << " votes=" << best_votes
            << " stable_frames=" << stable_frames
            << " frames_from_first_read=" << frames_from_first_read
            << " ms_from_first_read=" << ms_from_first_read
            << " frames_from_readable=" << frames_from_readable
            << " ms_from_readable=" << ms_from_readable
            << std::endl;
}

static std::string format_probability(float value) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(2) << clamp_probability(value);
  return out.str();
}

static std::string summarize_attribute_vote_map(const std::map<std::string, int>& vote_map,
                                                size_t max_entries = 6) {
  std::vector<std::pair<int, std::string>> ranked_votes;
  ranked_votes.reserve(vote_map.size());

  for (std::map<std::string, int>::const_iterator it = vote_map.begin();
       it != vote_map.end();
       ++it) {
    ranked_votes.push_back(std::make_pair(it->second, it->first));
  }

  std::sort(ranked_votes.begin(), ranked_votes.end(),
            [](const std::pair<int, std::string>& left,
               const std::pair<int, std::string>& right) {
              if (left.first != right.first) {
                return left.first > right.first;
              }
              return left.second < right.second;
            });

  std::ostringstream summary;
  for (size_t index = 0; index < ranked_votes.size() && index < max_entries; ++index) {
    if (index > 0) {
      summary << ", ";
    }
    summary << ranked_votes[index].second << ':' << ranked_votes[index].first;
  }

  return summary.str();
}

static std::string summarize_classifier_candidates(NvDsObjectMeta* obj_meta,
                                                   guint classifier_uid,
                                                   size_t max_entries = 6) {
  std::vector<std::pair<float, std::string>> ranked_candidates;

  if (!obj_meta) {
    return std::string();
  }

  for (NvDsMetaList* classifier_list = obj_meta->classifier_meta_list;
       classifier_list != NULL;
       classifier_list = classifier_list->next) {
    NvDsClassifierMeta* classifier_meta =
        static_cast<NvDsClassifierMeta*>(classifier_list->data);
    if (!classifier_meta ||
        static_cast<guint>(classifier_meta->unique_component_id) != classifier_uid) {
      continue;
    }

    for (NvDsMetaList* label_list = classifier_meta->label_info_list;
         label_list != NULL;
         label_list = label_list->next) {
      NvDsLabelInfo* label_info = static_cast<NvDsLabelInfo*>(label_list->data);
      if (!label_info || !label_info->result_label) {
        continue;
      }

      std::string label = normalize_vehicle_attribute_label(label_info->result_label,
                                                            classifier_uid);
      if (label.empty()) {
        continue;
      }

      float probability = clamp_probability(label_info->result_prob);
      std::ostringstream candidate;
      candidate << label << '@' << format_probability(probability);
      ranked_candidates.push_back(std::make_pair(probability, candidate.str()));
    }
  }

  std::sort(ranked_candidates.begin(), ranked_candidates.end(),
            [](const std::pair<float, std::string>& left,
               const std::pair<float, std::string>& right) {
              if (std::abs(left.first - right.first) > 1e-6f) {
                return left.first > right.first;
              }
              return left.second < right.second;
            });

  std::ostringstream summary;
  for (size_t index = 0;
       index < ranked_candidates.size() && index < max_entries;
       ++index) {
    if (index > 0) {
      summary << ", ";
    }
    summary << ranked_candidates[index].second;
  }

  return summary.str();
}

static bool is_large_vehicle_type_label(const std::string& label) {
  return label == "Truck" || label == "Large Vehicle";
}

static bool is_pickup_focused_large_vehicle_make(const std::string& label) {
  return label == "RAM" || label == "Ford" || label == "Chevrolet" ||
         label == "GMC" || label == "Toyota" || label == "Nissan" ||
         label == "Honda" || label == "Isuzu" || label == "Mazda" ||
         label == "Mitsubishi";
}

static bool prefer_pickup_large_vehicle_make(const PlateTrack& track,
                                             const VehicleAttributeObservations& observations) {
  int truck_votes = get_vote_count_for_label(track.type_votes, "Truck");
  int large_vehicle_votes = get_vote_count_for_label(track.type_votes, "Large Vehicle");
  return observations.type.label == "Truck" ||
         truck_votes >= std::max(4, large_vehicle_votes);
}

static float minimum_large_vehicle_make_probability(const std::string& label,
                                                    bool prefer_pickup) {
  if (label == "RAM") {
    return prefer_pickup ? 0.30f : 0.36f;
  }
  if (label == "Ford" || label == "Chevrolet" || label == "GMC" ||
      label == "Toyota" || label == "Nissan") {
    return prefer_pickup ? 0.38f : 0.46f;
  }
  if (label == "Honda" || label == "Isuzu" || label == "Mazda" ||
      label == "Mitsubishi") {
    return prefer_pickup ? 0.44f : 0.50f;
  }
  if (label == "Jeep") {
    return prefer_pickup ? 0.72f : 0.62f;
  }
  if (label == "Mercedes-Benz") {
    return prefer_pickup ? 0.74f : 0.64f;
  }
  return prefer_pickup ? 0.62f : 0.56f;
}

static double large_vehicle_make_priority(const std::string& label,
                                          bool prefer_pickup) {
  if (label == "RAM") {
    return prefer_pickup ? 1.50 : 1.28;
  }
  if (label == "Ford" || label == "Chevrolet" || label == "GMC") {
    return prefer_pickup ? 1.28 : 1.12;
  }
  if (label == "Toyota" || label == "Nissan") {
    return prefer_pickup ? 1.18 : 1.06;
  }
  if (label == "Honda" || label == "Isuzu" || label == "Mazda" ||
      label == "Mitsubishi") {
    return prefer_pickup ? 1.05 : 0.98;
  }
  if (label == "Jeep") {
    return prefer_pickup ? 0.72 : 0.90;
  }
  if (label == "Mercedes-Benz") {
    return prefer_pickup ? 0.66 : 0.88;
  }
  return prefer_pickup ? 0.80 : 0.92;
}

static ClassifierLabelResult get_best_plausible_large_vehicle_make_result(
    NvDsObjectMeta* obj_meta,
    bool prefer_pickup) {
  ClassifierLabelResult best_result;
  double best_score = 0.0;

  if (!obj_meta) {
    return best_result;
  }

  for (NvDsMetaList* classifier_list = obj_meta->classifier_meta_list;
       classifier_list != NULL;
       classifier_list = classifier_list->next) {
    NvDsClassifierMeta* classifier_meta =
        static_cast<NvDsClassifierMeta*>(classifier_list->data);
    if (!classifier_meta ||
        static_cast<guint>(classifier_meta->unique_component_id) !=
            VEHICLE_MAKE_CLASSIFIER_UID) {
      continue;
    }

    for (NvDsMetaList* label_list = classifier_meta->label_info_list;
         label_list != NULL;
         label_list = label_list->next) {
      NvDsLabelInfo* label_info = static_cast<NvDsLabelInfo*>(label_list->data);
      if (!label_info || !label_info->result_label) {
        continue;
      }

      std::string label = canonical_vehicle_make_for_large_vehicle(
          normalize_vehicle_attribute_label(label_info->result_label,
                                            VEHICLE_MAKE_CLASSIFIER_UID));
      if (!is_plausible_large_vehicle_make(label)) {
        continue;
      }

      float probability = clamp_probability(label_info->result_prob);
      if (probability < minimum_large_vehicle_make_probability(label, prefer_pickup)) {
        continue;
      }

      double score = probability * large_vehicle_make_priority(label, prefer_pickup);
      if (label == "RAM") {
        score += 0.08;
      }

      if (score > best_score ||
          (std::abs(score - best_score) < 1e-6 && probability > best_result.probability)) {
        best_result.label = label;
        best_result.probability = probability;
        best_score = score;
      }
    }
  }

  return best_result;
}

static std::string summarize_large_vehicle_make_candidates(NvDsObjectMeta* obj_meta,
                                                           bool prefer_pickup) {
  std::vector<std::pair<double, std::string>> ranked_candidates;

  if (!obj_meta) {
    return std::string();
  }

  for (NvDsMetaList* classifier_list = obj_meta->classifier_meta_list;
       classifier_list != NULL;
       classifier_list = classifier_list->next) {
    NvDsClassifierMeta* classifier_meta =
        static_cast<NvDsClassifierMeta*>(classifier_list->data);
    if (!classifier_meta ||
        static_cast<guint>(classifier_meta->unique_component_id) !=
            VEHICLE_MAKE_CLASSIFIER_UID) {
      continue;
    }

    for (NvDsMetaList* label_list = classifier_meta->label_info_list;
         label_list != NULL;
         label_list = label_list->next) {
      NvDsLabelInfo* label_info = static_cast<NvDsLabelInfo*>(label_list->data);
      if (!label_info || !label_info->result_label) {
        continue;
      }

      std::string normalized_label = normalize_vehicle_attribute_label(
          label_info->result_label, VEHICLE_MAKE_CLASSIFIER_UID);
      if (normalized_label.empty()) {
        continue;
      }

      std::string canonical_label =
          canonical_vehicle_make_for_large_vehicle(normalized_label);
      float probability = clamp_probability(label_info->result_prob);
      bool plausible = is_plausible_large_vehicle_make(canonical_label);
      double score = probability;

      if (plausible) {
        score = probability * large_vehicle_make_priority(canonical_label, prefer_pickup);
        if (canonical_label == "RAM") {
          score += 0.08;
        }
      }

      std::ostringstream candidate;
      candidate << canonical_label << '@' << format_probability(probability);
      if (!plausible) {
        candidate << "(filtered)";
      } else {
        candidate << "[min="
                  << format_probability(minimum_large_vehicle_make_probability(
                         canonical_label, prefer_pickup))
                  << ",score=" << std::fixed << std::setprecision(2) << score << ']';
      }
      ranked_candidates.push_back(std::make_pair(score, candidate.str()));
    }
  }

  std::sort(ranked_candidates.begin(), ranked_candidates.end(),
            [](const std::pair<double, std::string>& left,
               const std::pair<double, std::string>& right) {
              if (std::abs(left.first - right.first) > 1e-6) {
                return left.first > right.first;
              }
              return left.second < right.second;
            });

  std::ostringstream summary;
  for (size_t index = 0; index < ranked_candidates.size() && index < 8; ++index) {
    if (index > 0) {
      summary << ", ";
    }
    summary << ranked_candidates[index].second;
  }

  return summary.str();
}

static std::string canonical_vehicle_make_for_large_vehicle(const std::string& label) {
  std::string normalized = trim_copy(label);
  if (normalized.empty()) {
    return normalized;
  }

  std::string normalized_key = to_lower_copy(normalized);
  if (normalized_key == "dodge" || normalized_key == "ram" ||
      normalized_key == "ram trucks" || normalized_key == "dodge ram") {
    return "RAM";
  }
  if (normalized_key == "chevy") {
    return "Chevrolet";
  }
  return normalized;
}

static bool is_plausible_large_vehicle_make(const std::string& label) {
  std::string canonical_label = canonical_vehicle_make_for_large_vehicle(label);
  return canonical_label == "RAM" || canonical_label == "Ford" ||
         canonical_label == "Chevrolet" || canonical_label == "GMC" ||
         canonical_label == "Toyota" || canonical_label == "Nissan" ||
         canonical_label == "Honda" || canonical_label == "Jeep" ||
         canonical_label == "Isuzu" || canonical_label == "Mazda" ||
         canonical_label == "Mitsubishi" ||
         canonical_label == "Mercedes-Benz";
}

static bool is_strong_large_vehicle_type_observation(
    const ClassifierLabelResult& result) {
  return is_large_vehicle_type_label(result.label) &&
         clamp_probability(result.probability) >= 0.72f;
}

static bool is_strong_plausible_large_vehicle_make_observation(
    const ClassifierLabelResult& result) {
  std::string canonical_label =
      canonical_vehicle_make_for_large_vehicle(result.label);
  return is_plausible_large_vehicle_make(canonical_label) &&
         clamp_probability(result.probability) >= 0.72f;
}

static void remember_large_vehicle_track_hints(
    PlateTrack& track,
    const VehicleAttributeObservations& observations) {
  if (is_strong_large_vehicle_type_observation(observations.type)) {
    track.large_vehicle_hint_score = std::min(track.large_vehicle_hint_score + 2, 12);
    if (clamp_probability(observations.type.probability) >=
        clamp_probability(track.hinted_large_vehicle_type_probability)) {
      track.hinted_large_vehicle_type = observations.type.label;
      track.hinted_large_vehicle_type_probability =
          clamp_probability(observations.type.probability);
    }
  }

  if (is_strong_plausible_large_vehicle_make_observation(observations.make)) {
    track.large_vehicle_hint_score = std::min(track.large_vehicle_hint_score + 1, 12);
    if (clamp_probability(observations.make.probability) >=
        clamp_probability(track.hinted_large_vehicle_make_probability)) {
      track.hinted_large_vehicle_make =
          canonical_vehicle_make_for_large_vehicle(observations.make.label);
      track.hinted_large_vehicle_make_probability =
          clamp_probability(observations.make.probability);
    }
  }
}

static bool is_likely_large_vehicle_track(const PlateTrack& track,
                                          const VehicleAttributeObservations& observations) {
  int large_vehicle_type_votes = get_vote_count_for_label(track.type_votes, "Truck") +
                                 get_vote_count_for_label(track.type_votes, "Large Vehicle");
  return is_large_vehicle_type_label(observations.type.label) ||
         large_vehicle_type_votes >= 8 ||
         track.large_vehicle_hint_score >= 2 ||
         is_strong_plausible_large_vehicle_make_observation(observations.make);
}

static StabilizedAttributeResult choose_plausible_large_vehicle_make(
    const std::map<std::string, int>& vote_map) {
  std::map<std::string, int> plausible_votes;
  for (std::map<std::string, int>::const_iterator it = vote_map.begin();
       it != vote_map.end();
       ++it) {
    std::string canonical_label = canonical_vehicle_make_for_large_vehicle(it->first);
    if (!is_plausible_large_vehicle_make(canonical_label)) {
      continue;
    }
    plausible_votes[canonical_label] += it->second;
  }

  StabilizedAttributeResult result;
  for (std::map<std::string, int>::const_iterator it = plausible_votes.begin();
       it != plausible_votes.end();
       ++it) {
    result.total_votes += it->second;
    if (it->second > result.best_votes) {
      result.second_votes = result.best_votes;
      result.best_votes = it->second;
      result.label = it->first;
    } else if (it->second > result.second_votes) {
      result.second_votes = it->second;
    }
  }

  if (result.label.empty() || result.best_votes < 6) {
    return StabilizedAttributeResult{};
  }

  if (result.total_votes > 0 &&
      static_cast<double>(result.best_votes) / result.total_votes < 0.55) {
    return StabilizedAttributeResult{};
  }

  if (result.second_votes > 0 && result.best_votes < result.second_votes + 2) {
    return StabilizedAttributeResult{};
  }

  return result;
}

static std::string vehicle_type_family_for_label(const std::string& label) {
  if (is_large_vehicle_type_label(label)) {
    return "Large Vehicle";
  }
  return label;
}

static std::string canonical_vehicle_type_for_family(
    const std::string& family,
    const std::map<std::string, int>& vote_map) {
  if (family != "Large Vehicle") {
    return family;
  }

  int truck_votes = get_vote_count_for_label(vote_map, "Truck");
  int large_vehicle_votes = get_vote_count_for_label(vote_map, "Large Vehicle");
  if (truck_votes >= large_vehicle_votes + 2) {
    return "Truck";
  }
  return "Large Vehicle";
}

static std::string vehicle_color_family_for_label(const std::string& label) {
  if (label == "Maroon" || label == "Red") {
    return "Red";
  }
  if (label == "Silver" || label == "Gray") {
    return "Gray";
  }
  if (label == "Gold" || label == "Brown") {
    return "Brown";
  }
  return label;
}

static std::string canonical_vehicle_color_for_family(
    const std::string& family,
    const std::map<std::string, int>& vote_map) {
  if (family == "Red") {
    return "Red";
  }
  if (family == "Gray") {
    int silver_votes = get_vote_count_for_label(vote_map, "Silver");
    int gray_votes = get_vote_count_for_label(vote_map, "Gray");
    if (silver_votes >= gray_votes + 5) {
      return "Silver";
    }
    return "Gray";
  }
  if (family == "Brown") {
    int gold_votes = get_vote_count_for_label(vote_map, "Gold");
    int brown_votes = get_vote_count_for_label(vote_map, "Brown");
    if (gold_votes >= brown_votes + 4) {
      return "Gold";
    }
    return "Brown";
  }
  return family;
}

static bool is_infrared_tinted_vehicle_crop(const cv::Mat& crop) {
  if (crop.empty()) {
    return false;
  }

  std::vector<cv::Mat> channels;
  cv::split(crop, channels);
  if (channels.size() != 3) {
    return false;
  }

  cv::Mat max_channel;
  cv::Mat min_channel;
  cv::max(channels[0], channels[1], max_channel);
  cv::max(max_channel, channels[2], max_channel);
  cv::min(channels[0], channels[1], min_channel);
  cv::min(min_channel, channels[2], min_channel);

  cv::Mat spread = max_channel - min_channel;
  cv::Scalar mean_bgr = cv::mean(crop);
  double mean_spread = cv::mean(spread)[0];

  return mean_bgr[2] >= mean_bgr[1] + 10.0 &&
         mean_bgr[1] >= mean_bgr[0] + 3.0 &&
         mean_spread <= 38.0;
}

static cv::Rect compute_vehicle_body_rect(const cv::Mat& frame,
                                          const NvDsObjectMeta* vehicle_meta) {
  if (!vehicle_meta || frame.empty()) {
    return cv::Rect();
  }

  cv::Rect vehicle_rect = clamp_rect_to_frame(
      frame,
      static_cast<int>(vehicle_meta->rect_params.left),
      static_cast<int>(vehicle_meta->rect_params.top),
      static_cast<int>(vehicle_meta->rect_params.width),
      static_cast<int>(vehicle_meta->rect_params.height));
  if (vehicle_rect.width < 60 || vehicle_rect.height < 60) {
    return cv::Rect();
  }

  int inset_x = std::max(6, vehicle_rect.width / 9);
  int inset_top = std::max(8, vehicle_rect.height / 6);
  int inset_bottom = std::max(6, vehicle_rect.height / 10);
  cv::Rect body_rect(vehicle_rect.x + inset_x,
                     vehicle_rect.y + inset_top,
                     vehicle_rect.width - (inset_x * 2),
                     vehicle_rect.height - inset_top - inset_bottom);
  body_rect = clamp_rect_to_frame(frame,
                                  body_rect.x,
                                  body_rect.y,
                                  body_rect.width,
                                  body_rect.height);
  if (body_rect.width < 40 || body_rect.height < 30) {
    body_rect = vehicle_rect;
  }

  return body_rect;
}

static VehicleCropQuality assess_vehicle_crop_quality(const cv::Mat& frame,
                                                      const NvDsObjectMeta* vehicle_meta) {
  VehicleCropQuality quality;
  if (!vehicle_meta || frame.empty()) {
    return quality;
  }

  cv::Rect vehicle_rect = clamp_rect_to_frame(
      frame,
      static_cast<int>(vehicle_meta->rect_params.left),
      static_cast<int>(vehicle_meta->rect_params.top),
      static_cast<int>(vehicle_meta->rect_params.width),
      static_cast<int>(vehicle_meta->rect_params.height));
  cv::Rect body_rect = compute_vehicle_body_rect(frame, vehicle_meta);
  if (vehicle_rect.width <= 0 || vehicle_rect.height <= 0 ||
      body_rect.width <= 0 || body_rect.height <= 0) {
    return quality;
  }

  quality.valid = true;
  quality.clipped = vehicle_rect.x <= 4 ||
                    vehicle_rect.y <= 4 ||
                    (vehicle_rect.x + vehicle_rect.width) >= (frame.cols - 4) ||
                    (vehicle_rect.y + vehicle_rect.height) >= (frame.rows - 4);

  cv::Mat crop = frame(body_rect);
  if (crop.empty()) {
    quality.valid = false;
    return quality;
  }

  quality.likely_infrared = is_infrared_tinted_vehicle_crop(crop);

  cv::Mat hsv;
  cv::cvtColor(crop, hsv, cv::COLOR_BGR2HSV);

  int pixel_count = hsv.rows * hsv.cols;
  if (pixel_count <= 0) {
    quality.valid = false;
    return quality;
  }

  int overexposed_pixels = 0;
  int underexposed_pixels = 0;
  std::vector<cv::Mat> hsv_channels;
  cv::split(hsv, hsv_channels);
  cv::Scalar value_mean;
  cv::Scalar value_stddev;
  cv::meanStdDev(hsv_channels[2], value_mean, value_stddev);

  for (int y = 0; y < hsv.rows; ++y) {
    for (int x = 0; x < hsv.cols; ++x) {
      int value = hsv.at<cv::Vec3b>(y, x)[2];
      if (value >= 238) {
        ++overexposed_pixels;
      }
      if (value <= 28) {
        ++underexposed_pixels;
      }
    }
  }

  double overexposed_share = static_cast<double>(overexposed_pixels) / pixel_count;
  double underexposed_share = static_cast<double>(underexposed_pixels) / pixel_count;
  double contrast = value_stddev[0];

  quality.heavy_glare = overexposed_share >= 0.16;
  quality.heavy_shadow = underexposed_share >= 0.34;
  quality.low_contrast = contrast < (quality.likely_infrared ? 22.0 : 26.0);

  float score = 1.0f;
  if (quality.clipped) {
    score -= 0.22f;
  }
  if (quality.heavy_glare) {
    score -= 0.22f;
  }
  if (quality.heavy_shadow) {
    score -= 0.20f;
  }
  if (quality.low_contrast) {
    score -= 0.18f;
  }
  if (quality.likely_infrared) {
    score -= 0.12f;
  }
  if (overexposed_share >= 0.08 && underexposed_share >= 0.22) {
    score -= 0.10f;
  }

  quality.score = std::max(0.0f, std::min(score, 1.0f));
  return quality;
}

static cv::Rect clamp_rect_to_frame(const cv::Mat& frame,
                                    int left,
                                    int top,
                                    int width,
                                    int height) {
  if (frame.empty() || width <= 0 || height <= 0) {
    return cv::Rect();
  }

  int bounded_left = std::max(0, left);
  int bounded_top = std::max(0, top);
  int bounded_right = std::min(frame.cols, left + width);
  int bounded_bottom = std::min(frame.rows, top + height);

  if (bounded_right <= bounded_left || bounded_bottom <= bounded_top) {
    return cv::Rect();
  }

  return cv::Rect(bounded_left,
                  bounded_top,
                  bounded_right - bounded_left,
                  bounded_bottom - bounded_top);
}

static bool is_achromatic_color_label(const std::string& label) {
  return label == "White" || label == "Silver" || label == "Gray" ||
         label == "Black";
}

static std::string classify_chromatic_vehicle_color(int hue, int sat, int val) {
  if (sat < 45 || val < 20) {
    return std::string();
  }

  if (hue < 8 || hue >= 172) {
    return (val < 105) ? "Maroon" : "Red";
  }
  if (hue < 18) {
    return (val < 145) ? "Brown" : "Orange";
  }
  if (hue < 28) {
    return (val < 150 || sat < 100) ? "Gold" : "Yellow";
  }
  if (hue < 38) {
    return "Yellow";
  }
  if (hue < 90) {
    return "Green";
  }
  if (hue < 140) {
    return "Blue";
  }
  if (hue < 172) {
    return "Maroon";
  }

  return std::string();
}

static ClassifierLabelResult estimate_visual_vehicle_color(const cv::Mat& frame,
                                                           const NvDsObjectMeta* vehicle_meta,
                                                           const NvDsObjectMeta* plate_meta,
                                                           const VehicleCropQuality& crop_quality) {
  ClassifierLabelResult result;
  if (!vehicle_meta || frame.empty()) {
    return result;
  }

  cv::Rect body_rect = compute_vehicle_body_rect(frame, vehicle_meta);
  if (body_rect.width <= 0 || body_rect.height <= 0) {
    return result;
  }

  if (crop_quality.valid && crop_quality.score < 0.42f) {
    return result;
  }

  cv::Mat crop = frame(body_rect);
  if (crop.empty()) {
    return result;
  }

  bool likely_infrared = crop_quality.valid
                             ? crop_quality.likely_infrared
                             : is_infrared_tinted_vehicle_crop(crop);

  cv::Mat hsv;
  cv::cvtColor(crop, hsv, cv::COLOR_BGR2HSV);

  cv::Rect plate_rect_local;
  bool has_plate_rect = false;
  if (plate_meta) {
    cv::Rect plate_rect = clamp_rect_to_frame(
        frame,
        static_cast<int>(plate_meta->rect_params.left),
        static_cast<int>(plate_meta->rect_params.top),
        static_cast<int>(plate_meta->rect_params.width),
        static_cast<int>(plate_meta->rect_params.height));
    cv::Rect overlap = plate_rect & body_rect;
    if (overlap.width > 0 && overlap.height > 0) {
      int margin_x = std::max(4, overlap.width / 3);
      int margin_y = std::max(4, overlap.height / 2);
      plate_rect_local = cv::Rect(overlap.x - body_rect.x - margin_x,
                                  overlap.y - body_rect.y - margin_y,
                                  overlap.width + (margin_x * 2),
                                  overlap.height + (margin_y * 2));
      plate_rect_local = clamp_rect_to_frame(crop,
                                             plate_rect_local.x,
                                             plate_rect_local.y,
                                             plate_rect_local.width,
                                             plate_rect_local.height);
      has_plate_rect = plate_rect_local.width > 0 && plate_rect_local.height > 0;
    }
  }

  std::map<std::string, int> chroma_votes;
  std::vector<int> value_hist(256, 0);
  std::vector<int> sat_hist(256, 0);
  int valid_pixels = 0;
  int achromatic_pixels = 0;
  int chroma_pixels = 0;

  for (int y = 0; y < hsv.rows; ++y) {
    double y_ratio = static_cast<double>(y) / std::max(1, hsv.rows - 1);
    bool lower_center_band = y_ratio >= 0.48 && y_ratio <= 0.84;
    bool mid_body_band = y_ratio >= 0.30 && y_ratio <= 0.68;

    for (int x = 0; x < hsv.cols; ++x) {
      double x_ratio = static_cast<double>(x) / std::max(1, hsv.cols - 1);
      bool center_body = lower_center_band && x_ratio >= 0.14 && x_ratio <= 0.86;
      bool side_panels = mid_body_band && (x_ratio <= 0.26 || x_ratio >= 0.74);
      bool quarter_panels = mid_body_band && x_ratio >= 0.32 && x_ratio <= 0.68 &&
                            y_ratio >= 0.36 && y_ratio <= 0.58;
      if (!center_body && !side_panels && !quarter_panels) {
        continue;
      }

      if (has_plate_rect && plate_rect_local.contains(cv::Point(x, y))) {
        continue;
      }

      const cv::Vec3b pixel = hsv.at<cv::Vec3b>(y, x);
      int hue = pixel[0];
      int sat = pixel[1];
      int val = pixel[2];

      if (val <= 16) {
        continue;
      }
      if (val >= 245 && sat <= 42) {
        continue;
      }
      if (y_ratio <= 0.40 && sat <= 28 && val <= 110) {
        continue;
      }

      ++valid_pixels;
      value_hist[val]++;
      sat_hist[sat]++;

      if ((likely_infrared && sat <= 78) ||
          (!likely_infrared && (sat <= 46 || (sat <= 58 && val <= 150)))) {
        ++achromatic_pixels;
        continue;
      }

      std::string chroma_label = classify_chromatic_vehicle_color(hue, sat, val);
      if (!chroma_label.empty()) {
        chroma_votes[chroma_label] += 1;
        ++chroma_pixels;
      }
    }
  }

  if (valid_pixels < 180) {
    return result;
  }

  auto histogram_percentile = [](const std::vector<int>& histogram, double percentile) {
    int total = 0;
    for (int count : histogram) {
      total += count;
    }
    if (total <= 0) {
      return 0;
    }

    int threshold = static_cast<int>(std::ceil(percentile * total));
    int cumulative = 0;
    for (size_t index = 0; index < histogram.size(); ++index) {
      cumulative += histogram[index];
      if (cumulative >= threshold) {
        return static_cast<int>(index);
      }
    }

    return static_cast<int>(histogram.size() - 1);
  };

  int value_median = histogram_percentile(value_hist, 0.50);
  int value_p80 = histogram_percentile(value_hist, 0.80);
  int sat_p75 = histogram_percentile(sat_hist, 0.75);
  double achromatic_share = static_cast<double>(achromatic_pixels) /
                            static_cast<double>(std::max(1, valid_pixels));

  std::string best_chroma_label;
  int best_chroma_votes = 0;
  int second_chroma_votes = 0;
  for (const auto& kv : chroma_votes) {
    if (kv.second > best_chroma_votes) {
      second_chroma_votes = best_chroma_votes;
      best_chroma_votes = kv.second;
      best_chroma_label = kv.first;
    } else if (kv.second > second_chroma_votes) {
      second_chroma_votes = kv.second;
    }
  }

  if (!best_chroma_label.empty() &&
      chroma_pixels >= (likely_infrared ? 140 : 90)) {
    double dominant_share = static_cast<double>(best_chroma_votes) /
                            static_cast<double>(std::max(1, chroma_pixels));
    if (!crop_quality.heavy_glare &&
        !crop_quality.heavy_shadow &&
        dominant_share >= (likely_infrared ? 0.58 : 0.42) &&
        best_chroma_votes >= second_chroma_votes + (likely_infrared ? 34 : 18) &&
        best_chroma_votes >= std::max(likely_infrared ? 80 : 50, valid_pixels / 10)) {
      result.label = best_chroma_label;
      result.probability = static_cast<float>(std::min(
          0.92,
          (likely_infrared ? 0.58 : 0.68) +
              (dominant_share * (likely_infrared ? 0.20 : 0.24)) -
              (achromatic_share * (likely_infrared ? 0.14 : 0.08))));
      return result;
    }
  }

  if (achromatic_share < 0.50 && best_chroma_label.empty()) {
    return result;
  }

  bool strongly_achromatic = achromatic_share >= 0.68 || sat_p75 <= 60;
  if (!strongly_achromatic && best_chroma_label.empty()) {
    return result;
  }

  if (value_median >= 182 || value_p80 >= 228) {
    result.label = "White";
    result.probability = 0.82f;
  } else if (value_median >= 142) {
    result.label = "Silver";
    result.probability = 0.80f;
  } else if (value_median >= 82) {
    result.label = "Gray";
    result.probability = 0.84f;
  } else {
    result.label = "Black";
    result.probability = 0.86f;
  }

  if (achromatic_share >= 0.80) {
    result.probability = std::min(0.92f, result.probability + 0.04f);
  }

  return result;
}

static bool vehicle_meta_meets_attribute_size(const NvDsObjectMeta* vehicle_meta,
                                              guint classifier_uid) {
  if (!vehicle_meta) {
    return false;
  }

  float width = vehicle_meta->rect_params.width;
  float height = vehicle_meta->rect_params.height;

  switch (classifier_uid) {
    case VEHICLE_MAKE_CLASSIFIER_UID:
      return width >= 260.0f && height >= 180.0f;
    case VEHICLE_TYPE_CLASSIFIER_UID:
      return width >= 180.0f && height >= 120.0f;
    case VEHICLE_COLOR_CLASSIFIER_UID:
      return width >= 180.0f && height >= 120.0f;
    default:
      return false;
  }
}

static VehicleAttributeVoteContext build_vehicle_attribute_vote_context(
    const NvDsObjectMeta* vehicle_meta,
    const std::string& consensus_plate,
    const PlateConfidenceResult& conf,
    int best_votes,
  int stable_frames,
  const VehicleCropQuality& crop_quality,
  bool likely_large_vehicle) {
  VehicleAttributeVoteContext context;

  if (!vehicle_meta || consensus_plate.empty() ||
      consensus_plate.find('?') != std::string::npos ||
      !is_valid_plate_text(consensus_plate)) {
    return context;
  }

  bool confident_plate = conf.score >= DEBUG_THRESHOLD_MIN;
  bool stable_plate = best_votes >= 20 && stable_frames >= 2;
  if (!confident_plate || !stable_plate) {
    return context;
  }

  if (!crop_quality.valid || crop_quality.score < 0.32f) {
    return context;
  }

  context.allow_type = vehicle_meta_meets_attribute_size(vehicle_meta, VEHICLE_TYPE_CLASSIFIER_UID) &&
                       conf.score >= REJECT_THRESHOLD &&
                       best_votes >= 24 &&
                       stable_frames >= 4 &&
                       !crop_quality.clipped &&
                       !crop_quality.heavy_shadow &&
                       crop_quality.score >= (crop_quality.likely_infrared ? 0.70f : 0.58f);
  context.allow_color = vehicle_meta_meets_attribute_size(vehicle_meta, VEHICLE_COLOR_CLASSIFIER_UID) &&
                        best_votes >= 22 &&
                        stable_frames >= 3 &&
                        !crop_quality.clipped &&
                        crop_quality.score >= (crop_quality.likely_infrared ? 0.74f : 0.50f) &&
                        !(crop_quality.likely_infrared && crop_quality.heavy_glare);
    bool make_size_ok =
      vehicle_meta_meets_attribute_size(vehicle_meta, VEHICLE_MAKE_CLASSIFIER_UID);
    bool strict_make_gate = make_size_ok &&
                conf.score >= LOCK_THRESHOLD &&
                best_votes >= 34 &&
                stable_frames >= 6 &&
                !crop_quality.likely_infrared &&
                !crop_quality.clipped &&
                !crop_quality.heavy_glare &&
                !crop_quality.heavy_shadow &&
                !crop_quality.low_contrast &&
                crop_quality.score >= 0.80f;
    bool relaxed_large_vehicle_make_gate = likely_large_vehicle &&
                       make_size_ok &&
                       conf.score >= REJECT_THRESHOLD &&
                       best_votes >= 24 &&
                       stable_frames >= 3 &&
                       !crop_quality.clipped &&
                       !crop_quality.heavy_shadow &&
                       !crop_quality.low_contrast &&
                       crop_quality.score >=
                         (crop_quality.likely_infrared ? 0.58f : 0.50f) &&
                       !(crop_quality.likely_infrared && crop_quality.heavy_glare);
    context.allow_make = strict_make_gate || relaxed_large_vehicle_make_gate;
    context.relaxed_large_vehicle_make = relaxed_large_vehicle_make_gate && !strict_make_gate;

  if (conf.score >= LOCK_THRESHOLD) {
    context.type_bonus += 2;
    context.color_bonus += 2;
    context.make_bonus += 3;
  } else if (conf.score >= REJECT_THRESHOLD) {
    context.type_bonus += 1;
    context.color_bonus += 1;
    context.make_bonus += 2;
  }

  if (crop_quality.score >= 0.82f) {
    context.type_bonus += 1;
    context.color_bonus += 1;
  }

  if (crop_quality.score >= 0.90f && !crop_quality.likely_infrared) {
    context.make_bonus += 1;
  }

  if (context.relaxed_large_vehicle_make) {
    context.make_bonus = std::max(context.make_bonus, conf.score >= LOCK_THRESHOLD ? 2 : 1);
    if (crop_quality.score >= 0.70f) {
      context.make_bonus += 1;
    }
  }

  return context;
}

static void penalize_attribute_votes_for_bad_lighting(PlateTrack& track,
                                                      const VehicleCropQuality& crop_quality) {
  if (!crop_quality.valid) {
    return;
  }

  int decay_steps = 0;
  if (crop_quality.score < 0.32f) {
    decay_steps += 2;
  } else if (crop_quality.score < 0.45f) {
    decay_steps += 1;
  }

  if (crop_quality.clipped) {
    decay_steps += 1;
  }
  if (crop_quality.heavy_glare) {
    decay_steps += 1;
  }
  if (crop_quality.heavy_shadow) {
    decay_steps += 1;
  }
  if (crop_quality.likely_infrared && crop_quality.score < 0.68f) {
    decay_steps += 1;
  }

  if (decay_steps <= 0) {
    return;
  }

  for (int step = 0; step < decay_steps; ++step) {
    decay_attribute_votes(track.color_votes);
    decay_attribute_votes(track.type_votes);
    if (step > 0 || crop_quality.score < 0.40f) {
      decay_attribute_votes(track.make_votes);
    }
  }
}

static void decay_competing_attribute_votes(std::map<std::string, int>& vote_map,
                                            const std::string& winning_label,
                                            int decay_amount) {
  if (decay_amount <= 0) {
    return;
  }

  for (std::map<std::string, int>::iterator it = vote_map.begin(); it != vote_map.end();) {
    if (it->first != winning_label) {
      it->second = std::max(0, it->second - decay_amount);
    }

    if (it->second == 0) {
      it = vote_map.erase(it);
    } else {
      ++it;
    }
  }
}

static void add_attribute_vote(std::map<std::string, int>& vote_map,
                               guint classifier_uid,
                               const ClassifierLabelResult& result,
                               int extra_weight = 0,
                               float minimum_probability_override = -1.0f) {
  if (result.label.empty()) {
    return;
  }

  float clamped_probability = clamp_probability(result.probability);
  float minimum_probability = minimum_attribute_probability(classifier_uid);
  if (minimum_probability_override >= 0.0f) {
    minimum_probability = minimum_probability_override;
  }
  if (clamped_probability > 0.0f && clamped_probability < minimum_probability) {
    return;
  }

  int competitor_decay = 0;
  if (classifier_uid == VEHICLE_MAKE_CLASSIFIER_UID) {
    competitor_decay = clamped_probability >= 0.82f ? 2 : 1;
  } else if (classifier_uid == VEHICLE_TYPE_CLASSIFIER_UID) {
    competitor_decay = clamped_probability >= 0.72f ? 1 : 0;
  }
  decay_competing_attribute_votes(vote_map, result.label, competitor_decay);

  vote_map[result.label] += attribute_vote_weight(classifier_uid, clamped_probability) +
                            std::max(0, extra_weight);
}

static void add_vehicle_attribute_votes(NvDsObjectMeta* vehicle_meta,
                                        PlateTrack& track,
                                        const VehicleAttributeObservations& observations,
                                        const VehicleAttributeVoteContext& context) {
  bool likely_large_vehicle = is_likely_large_vehicle_track(track, observations);

  if (context.allow_make) {
    ClassifierLabelResult make_observation = observations.make;
    int make_bonus = context.make_bonus;
    float minimum_make_probability_override = -1.0f;

    if (likely_large_vehicle) {
      bool prefer_pickup = prefer_pickup_large_vehicle_make(track, observations);
      ClassifierLabelResult plausible_make =
          get_best_plausible_large_vehicle_make_result(vehicle_meta, prefer_pickup);

      if (!plausible_make.label.empty()) {
        make_observation = plausible_make;
      } else if (!track.hinted_large_vehicle_make.empty()) {
        make_observation.label = track.hinted_large_vehicle_make;
        make_observation.probability = std::max(
            clamp_probability(make_observation.probability),
            clamp_probability(track.hinted_large_vehicle_make_probability));
      } else {
        make_observation.label =
            canonical_vehicle_make_for_large_vehicle(make_observation.label);
      }

      if (!make_observation.label.empty() &&
          !is_plausible_large_vehicle_make(make_observation.label)) {
        make_observation.label.clear();
      } else if (!make_observation.label.empty()) {
        make_bonus += 2;
        minimum_make_probability_override = context.relaxed_large_vehicle_make ? 0.52f : 0.60f;
        if (prefer_pickup && !is_pickup_focused_large_vehicle_make(make_observation.label)) {
          minimum_make_probability_override = std::max(minimum_make_probability_override, 0.70f);
        }
        if (make_observation.label == "RAM") {
          make_bonus += 3;
          minimum_make_probability_override = prefer_pickup ? 0.30f : 0.40f;
        }
      }
    }

    add_attribute_vote(track.make_votes, VEHICLE_MAKE_CLASSIFIER_UID,
                       make_observation, make_bonus, minimum_make_probability_override);
  }
  if (context.allow_type) {
    ClassifierLabelResult type_observation = observations.type;
    int type_bonus = context.type_bonus;
    if (likely_large_vehicle && !is_large_vehicle_type_label(type_observation.label) &&
        !track.hinted_large_vehicle_type.empty()) {
      type_observation.label = track.hinted_large_vehicle_type;
      type_observation.probability = std::max(
          clamp_probability(type_observation.probability),
          clamp_probability(track.hinted_large_vehicle_type_probability));
      type_bonus += 2;
    }

    add_attribute_vote(track.type_votes, VEHICLE_TYPE_CLASSIFIER_UID,
                       type_observation, type_bonus);
  }
  if (context.allow_color) {
    add_attribute_vote(track.color_votes, VEHICLE_COLOR_CLASSIFIER_UID,
                       observations.color, context.color_bonus);
  }
}

static void add_visual_color_vote(PlateTrack& track,
                                  const ClassifierLabelResult& visual_color,
                                  const VehicleAttributeVoteContext& context) {
  if (!context.allow_color || visual_color.label.empty()) {
    return;
  }

  int visual_bonus = is_achromatic_color_label(visual_color.label) ? 3 : 5;
  add_attribute_vote(track.color_votes,
                     VEHICLE_COLOR_CLASSIFIER_UID,
                     visual_color,
                     context.color_bonus + visual_bonus);
}

static void decay_attribute_votes(std::map<std::string, int>& vote_map) {
  for (auto it = vote_map.begin(); it != vote_map.end();) {
    it->second = std::max(0, it->second - 1);
    if (it->second == 0) {
      it = vote_map.erase(it);
    } else {
      ++it;
    }
  }
}

static StabilizedAttributeResult choose_stable_attribute(
    const std::map<std::string, int>& vote_map,
    guint classifier_uid) {
  StabilizedAttributeResult result;

  for (const auto& kv : vote_map) {
    result.total_votes += kv.second;
    if (kv.second > result.best_votes) {
      result.second_votes = result.best_votes;
      result.best_votes = kv.second;
      result.label = kv.first;
    } else if (kv.second > result.second_votes) {
      result.second_votes = kv.second;
    }
  }

  if (result.label.empty()) {
    return StabilizedAttributeResult{};
  }

  int min_best_votes = 6;
  int min_gap = 2;
  double min_ratio = 1.2;
  double min_share = 0.4;

  if (classifier_uid == VEHICLE_MAKE_CLASSIFIER_UID) {
    min_best_votes = 18;
    min_gap = 8;
    min_ratio = 2.0;
    min_share = 0.72;
  } else if (classifier_uid == VEHICLE_TYPE_CLASSIFIER_UID) {
    min_best_votes = 12;
    min_gap = 5;
    min_ratio = 1.5;
    min_share = 0.62;
  } else if (classifier_uid == VEHICLE_COLOR_CLASSIFIER_UID) {
    min_best_votes = 10;
    min_gap = 4;
    min_ratio = 1.45;
    min_share = 0.55;
  }

  if (result.best_votes < min_best_votes) {
    return StabilizedAttributeResult{};
  }

  if (result.total_votes > 0 &&
      static_cast<double>(result.best_votes) / result.total_votes < min_share) {
    return StabilizedAttributeResult{};
  }

  if (result.second_votes > 0) {
    if ((result.best_votes - result.second_votes) < min_gap) {
      return StabilizedAttributeResult{};
    }

    if (static_cast<double>(result.best_votes) / result.second_votes < min_ratio) {
      return StabilizedAttributeResult{};
    }
  }

  return result;
}

static StabilizedAttributeResult choose_stable_family_attribute(
    const std::map<std::string, int>& vote_map,
    guint classifier_uid) {
  if (classifier_uid != VEHICLE_TYPE_CLASSIFIER_UID &&
      classifier_uid != VEHICLE_COLOR_CLASSIFIER_UID) {
    return choose_stable_attribute(vote_map, classifier_uid);
  }

  std::map<std::string, int> family_votes;
  for (std::map<std::string, int>::const_iterator it = vote_map.begin();
       it != vote_map.end();
       ++it) {
    std::string family = classifier_uid == VEHICLE_TYPE_CLASSIFIER_UID
                             ? vehicle_type_family_for_label(it->first)
                             : vehicle_color_family_for_label(it->first);
    family_votes[family] += it->second;
  }

  StabilizedAttributeResult result = choose_stable_attribute(family_votes, classifier_uid);
  if (result.label.empty()) {
    return result;
  }

  result.label = classifier_uid == VEHICLE_TYPE_CLASSIFIER_UID
                     ? canonical_vehicle_type_for_family(result.label, vote_map)
                     : canonical_vehicle_color_for_family(result.label, vote_map);
  return result;
}

static VehicleAttributes resolve_vehicle_attributes(const PlateTrack& track) {
  VehicleAttributes attributes;

  StabilizedAttributeResult make_result =
      choose_stable_attribute(track.make_votes, VEHICLE_MAKE_CLASSIFIER_UID);
  StabilizedAttributeResult type_result =
      choose_stable_family_attribute(track.type_votes, VEHICLE_TYPE_CLASSIFIER_UID);
  StabilizedAttributeResult color_result =
      choose_stable_family_attribute(track.color_votes, VEHICLE_COLOR_CLASSIFIER_UID);

  attributes.make = make_result.label;
  attributes.type = type_result.label;
  attributes.color = color_result.label;

  if (is_large_vehicle_type_label(attributes.type)) {
    attributes.make = canonical_vehicle_make_for_large_vehicle(attributes.make);
    if (!attributes.make.empty() &&
        !is_plausible_large_vehicle_make(attributes.make)) {
      attributes.make.clear();
    }

    if (attributes.make.empty()) {
      attributes.make = choose_plausible_large_vehicle_make(track.make_votes).label;
    }
  }

  if (!attributes.make.empty() && attributes.type.empty() && make_result.best_votes < 24) {
    attributes.make.clear();
  }

  if (attributes.color.empty() && (!attributes.type.empty() || !attributes.make.empty())) {
    attributes.color = "Unknown";
  }

  if (is_large_vehicle_type_label(attributes.type)) {
    attributes.make = canonical_vehicle_make_for_large_vehicle(attributes.make);
  }

  return attributes;
}

static bool has_meaningful_vehicle_attributes(const VehicleAttributes& attributes) {
  return !attributes.make.empty() || !attributes.type.empty() ||
         !attributes.color.empty();
}

static bool should_persist_recent_plate_attributes(const VehicleAttributes& attributes) {
  if (!has_meaningful_vehicle_attributes(attributes)) {
    return false;
  }

  if (!attributes.make.empty()) {
    return true;
  }

  return is_large_vehicle_type_label(attributes.type);
}

static const RecentPlateAttributes* lookup_recent_plate_attributes(
    const std::string& plate,
    const std::string& video_source,
    int current_frame) {
  std::string normalized_plate = normalize_plate(plate);
  if (normalized_plate.empty()) {
    return nullptr;
  }

  std::map<std::string, RecentPlateAttributes>::const_iterator it =
      recent_plate_attributes.find(normalized_plate);
  if (it == recent_plate_attributes.end()) {
    return nullptr;
  }

  if (!it->second.video_source.empty() && !video_source.empty() &&
      it->second.video_source != video_source) {
    return nullptr;
  }

  if ((current_frame - it->second.frame_number) >
      RECENT_PLATE_ATTRIBUTE_MAX_AGE_FRAMES) {
    return nullptr;
  }

  return &it->second;
}

static void apply_recent_plate_attribute_prior(
    const std::string& plate,
    const std::string& video_source,
    int current_frame,
    const VehicleCropQuality& crop_quality,
    const VehicleAttributeObservations& observations,
    VehicleAttributes& attributes) {
  if (!is_valid_plate_text(plate)) {
    return;
  }

  const RecentPlateAttributes* prior =
      lookup_recent_plate_attributes(plate, video_source, current_frame);
  if (!prior || !has_meaningful_vehicle_attributes(prior->attributes)) {
    return;
  }

  const VehicleAttributes& prior_attributes = prior->attributes;
  bool prior_large_vehicle = is_large_vehicle_type_label(prior_attributes.type);
  std::string canonical_observed_make =
      canonical_vehicle_make_for_large_vehicle(observations.make.label);
  bool strong_live_ram_signal =
      canonical_observed_make == "RAM" &&
      clamp_probability(observations.make.probability) >= 0.72f &&
      (is_large_vehicle_type_label(observations.type.label) ||
       is_large_vehicle_type_label(attributes.type) ||
       prior_large_vehicle);

  if (strong_live_ram_signal) {
    attributes.make = "RAM";
    if (attributes.type.empty() || attributes.type == "SUV") {
      if (is_large_vehicle_type_label(observations.type.label)) {
        attributes.type = observations.type.label;
      } else if (prior_large_vehicle) {
        attributes.type = prior_attributes.type;
      } else {
        attributes.type = "Truck";
      }
    }
  }

  bool missing_or_suv_type = attributes.type.empty() || attributes.type == "SUV";
  bool weak_make_signal = attributes.make.empty() &&
                          (observations.make.label.empty() ||
                           observations.make.probability < 0.55f);
  std::string canonical_current_make =
      canonical_vehicle_make_for_large_vehicle(attributes.make);
  bool conflicting_large_vehicle_make =
      prior_large_vehicle && !prior_attributes.make.empty() &&
      (!canonical_current_make.empty() && canonical_current_make != prior_attributes.make) &&
      (missing_or_suv_type || !is_plausible_large_vehicle_make(canonical_current_make));
  bool infrared_color_conflict = crop_quality.likely_infrared &&
                                 (attributes.color.empty() ||
                                  attributes.color == "Red" ||
                                  attributes.color == "Maroon");

  if (missing_or_suv_type && weak_make_signal &&
      prior_large_vehicle) {
    attributes.type = prior_attributes.type;
  }

  if (prior_large_vehicle && conflicting_large_vehicle_make) {
    attributes.type = prior_attributes.type;
  }

  if (!prior_attributes.make.empty() &&
      (attributes.make.empty() || conflicting_large_vehicle_make ||
       (prior_large_vehicle && missing_or_suv_type)) &&
      (is_large_vehicle_type_label(attributes.type) || missing_or_suv_type ||
       prior_large_vehicle)) {
    attributes.make = prior_attributes.make;
  }

  if (!prior_attributes.color.empty() &&
      (infrared_color_conflict ||
       (attributes.color.empty() && is_large_vehicle_type_label(attributes.type)))) {
    attributes.color = prior_attributes.color;
  }

  if (attributes.color.empty() && (!attributes.type.empty() || !attributes.make.empty())) {
    attributes.color = "Unknown";
  }
}

static bool has_supported_large_vehicle_make_snapshot(
    const PlateTrack& track,
    const VehicleAttributeObservations& observations,
    const std::string& make) {
  std::string canonical_make = canonical_vehicle_make_for_large_vehicle(make);
  if (canonical_make.empty() || !is_plausible_large_vehicle_make(canonical_make)) {
    return false;
  }

  StabilizedAttributeResult plausible_votes =
      choose_plausible_large_vehicle_make(track.make_votes);
  if (plausible_votes.label == canonical_make) {
    return true;
  }

  if (track.hinted_large_vehicle_make == canonical_make &&
      clamp_probability(track.hinted_large_vehicle_make_probability) >= 0.72f) {
    return true;
  }

  return canonical_vehicle_make_for_large_vehicle(observations.make.label) == canonical_make &&
         clamp_probability(observations.make.probability) >=
             (canonical_make == "RAM" ? 0.72f : 0.84f);
}

static void remember_recent_plate_attributes(const std::string& plate,
                                             const std::string& video_source,
                                             int current_frame,
                                             int confidence,
                                             const PlateTrack& track,
                                             const VehicleAttributeObservations& observations,
                                             const VehicleAttributes& attributes) {
  if (!is_valid_plate_text(plate) || confidence < REJECT_THRESHOLD) {
    return;
  }

  VehicleAttributes snapshot_attributes = attributes;
  if (is_large_vehicle_type_label(snapshot_attributes.type) &&
      !snapshot_attributes.make.empty() &&
      !has_supported_large_vehicle_make_snapshot(track,
                                                 observations,
                                                 snapshot_attributes.make)) {
    snapshot_attributes.make.clear();
  }

  if (!should_persist_recent_plate_attributes(snapshot_attributes)) {
    return;
  }

  std::string normalized_plate = normalize_plate(plate);
  if (normalized_plate.empty()) {
    return;
  }

  RecentPlateAttributes& snapshot = recent_plate_attributes[normalized_plate];
  if (snapshot.frame_number > current_frame) {
    return;
  }

  snapshot.attributes = snapshot_attributes;
  snapshot.video_source = video_source;
  snapshot.frame_number = current_frame;
  snapshot.confidence = confidence;
}

static bool is_valid_plate_text(const std::string& text) {
    if (text.length() < 4 || text.length() > 8)
        return false;

    for (char c : text) {
        if (!std::isalnum(static_cast<unsigned char>(c))) {
            return false;
        }
    }
    return true;
}

static bool has_letters_and_digits(const std::string& text) {
    bool has_letter = false;
    bool has_digit = false;

    for (char c : text) {
        if (std::isalpha(static_cast<unsigned char>(c))) has_letter = true;
        if (std::isdigit(static_cast<unsigned char>(c))) has_digit = true;
    }

    return has_letter && has_digit;
}

  static void decay_plate_track(PlateTrack& track, int current_frame) {
    int unseen = current_frame - track.last_seen_frame;
    track.missed_frames = unseen;

    if (unseen < TRACK_DECAY_FRAMES) return;

    for (auto it = track.votes.begin(); it != track.votes.end();) {
      it->second = std::max(0, it->second - 1);
      if (it->second == 0) it = track.votes.erase(it);
      else ++it;
    }

    for (auto pos_it = track.char_votes.begin(); pos_it != track.char_votes.end();) {
      for (auto ch_it = pos_it->second.begin(); ch_it != pos_it->second.end();) {
        ch_it->second = std::max(0, ch_it->second - 1);
        if (ch_it->second == 0) ch_it = pos_it->second.erase(ch_it);
        else ++ch_it;
      }

      if (pos_it->second.empty()) pos_it = track.char_votes.erase(pos_it);
      else ++pos_it;
    }

    decay_attribute_votes(track.make_votes);
    decay_attribute_votes(track.type_votes);
    decay_attribute_votes(track.color_votes);
    if (track.large_vehicle_hint_score > 0) {
      track.large_vehicle_hint_score = std::max(0, track.large_vehicle_hint_score - 1);
      if (track.large_vehicle_hint_score == 0) {
        track.hinted_large_vehicle_type.clear();
        track.hinted_large_vehicle_type_probability = 0.0f;
        track.hinted_large_vehicle_make.clear();
        track.hinted_large_vehicle_make_probability = 0.0f;
      }
    }

    if (track.locked && unseen > LOCK_HOLD_FRAMES) {
      track.locked = false;
      track.locked_plate.clear();
      track.locked_confidence = 0;
      track.lock_frame = -1;
    }

    if (!track.reported && unseen > TRACK_DROP_FRAMES / 2) {
      track.debug_reported = false;
    }
  }

  static bool should_lock_plate(const std::string& plate, int confidence, int stable_frames) {
    if (plate.empty()) return false;
    if (plate.find('?') != std::string::npos) return false;
    if (confidence < LOCK_THRESHOLD) return false;
    if (stable_frames < LOCK_STABLE_FRAMES) return false;
    return true;
  }

static bool should_emit_track_event(const std::string& current_plate,
                                    int current_confidence,
                                    const std::string& last_plate,
                                    int last_confidence,
                                    int last_frame,
                                    int current_frame) {
  if (current_plate.empty()) {
    return false;
  }
  if (last_frame < 0 || last_plate.empty()) {
    return true;
  }
  if (current_plate != last_plate) {
    return true;
  }
  if (current_confidence >= last_confidence + TRACK_EVENT_IMPROVEMENT_DELTA) {
    return true;
  }
  return (current_frame - last_frame) >= TRACK_EVENT_COOLDOWN_FRAMES;
}

static bool looks_like_reasonable_us_plate(const std::string& text) {
    // Tighten length: most useful reads will be 6-8 chars
    if (text.length() < 6 || text.length() > 8) return false;

    // Must be alphanumeric only
    for (char c : text) {
        if (!std::isalnum(static_cast<unsigned char>(c))) return false;
    }

    // Must contain both letters and digits
    if (!has_letters_and_digits(text)) return false;

    return true;
}

static int plate_vote_weight(const std::string& text, float classifier_prob = 1.0f) {
    int weight = 0;

    // Base preference for reasonable length
    weight += static_cast<int>(text.length());

    // Strong preference for 7-char plates
    if (text.length() == 7) weight += 4;
    else if (text.length() == 6 || text.length() == 8) weight += 2;

    // Must contain both letters and digits
    if (has_letters_and_digits(text)) weight += 3;

    // Extra bias toward common US/CA-looking structure:
    // starts with digit, has letters in middle, ends with digits
    if (text.length() == 7 &&
        std::isdigit(static_cast<unsigned char>(text[0])) &&
        std::isalpha(static_cast<unsigned char>(text[1])) &&
        std::isalpha(static_cast<unsigned char>(text[2])) &&
        std::isalpha(static_cast<unsigned char>(text[3])) &&
        std::isdigit(static_cast<unsigned char>(text[4])) &&
        std::isdigit(static_cast<unsigned char>(text[5])) &&
        std::isdigit(static_cast<unsigned char>(text[6]))) {
        weight += 6;
    }

    if (classifier_prob > 0.0f) {
      float clamped_prob = std::max(0.05f, std::min(classifier_prob, 1.0f));
      weight += static_cast<int>(clamped_prob * 8.0f) - 2;
    }

    if (weight < 1) {
      weight = 1;
    }

    return weight;
}

static int plate_distance(const std::string& a, const std::string& b) {
    if (std::abs((int)a.length() - (int)b.length()) > 1) return 999;

    // Simple distance for equal lengths
    if (a.length() == b.length()) {
        int diff = 0;
        for (size_t i = 0; i < a.length(); i++) {
            if (a[i] != b[i]) diff++;
        }
        return diff;
    }

    // Handle one missing/extra character
    const std::string& shorter = (a.length() < b.length()) ? a : b;
    const std::string& longer  = (a.length() < b.length()) ? b : a;

    size_t i = 0, j = 0;
    int diff = 0;

    while (i < shorter.length() && j < longer.length()) {
        if (shorter[i] == longer[j]) {
            i++;
            j++;
        } else {
            diff++;
            j++;
            if (diff > 1) return diff;
        }
    }

    if (j < longer.length()) diff++;

    return diff;
}

static bool same_plate_family(const std::string& a, const std::string& b) {
    return plate_distance(a, b) <= 2;
}

enum class PlateCharExpectation {
  Any,
  Letter,
  Digit,
};

// ---------------------------------------------------------------------------
// California plate pattern matching
// ---------------------------------------------------------------------------

enum class CAPlatePattern {
    // Standard passenger:  7LLL000  (digit, 3 letters, 3 digits)
    Passenger,
    // Commercial format A: LLL000A  (3 letters, 3 digits, 1 letter)
    Commercial1,
    // Commercial format B: 0L00000  (digit, letter, 5 digits)
    Commercial2,
    // Motorcycle:          00L000   (2 digits, letter, 3 digits)
    Motorcycle,
    // Trailer:             0LL000   (digit, 2 letters, 3 digits)
    Trailer,
    // Older 6-char:        LLL000   (3 letters, 3 digits)
    SixChar,
    Unknown
};

struct CAPatternFit {
    CAPlatePattern pattern = CAPlatePattern::Unknown;
    int fit_score = 0;   // 0..length  (one point per matching position)
    std::string name;    // human-readable label
};

static PlateCharExpectation ca_pattern_char_class(CAPlatePattern p, size_t pos) {
    switch (p) {
        case CAPlatePattern::Passenger:
            // 7LLL000  (pos 0 = digit, 1-3 = letter, 4-6 = digit)
            if (pos == 0 || pos >= 4) return PlateCharExpectation::Digit;
            return PlateCharExpectation::Letter;
        case CAPlatePattern::Commercial1:
            // LLL000A
            if (pos < 3) return PlateCharExpectation::Letter;
            if (pos < 6) return PlateCharExpectation::Digit;
            return PlateCharExpectation::Letter;  // pos 6
        case CAPlatePattern::Commercial2:
            // 0L00000
            if (pos == 0) return PlateCharExpectation::Digit;
            if (pos == 1) return PlateCharExpectation::Letter;
            return PlateCharExpectation::Digit;
        case CAPlatePattern::Motorcycle:
            // 00L000
            if (pos < 2) return PlateCharExpectation::Digit;
            if (pos == 2) return PlateCharExpectation::Letter;
            return PlateCharExpectation::Digit;
        case CAPlatePattern::Trailer:
            // 0LL000
            if (pos == 0) return PlateCharExpectation::Digit;
            if (pos < 3) return PlateCharExpectation::Letter;
            return PlateCharExpectation::Digit;
        case CAPlatePattern::SixChar:
            // LLL000
            if (pos < 3) return PlateCharExpectation::Letter;
            return PlateCharExpectation::Digit;
        default:
            return PlateCharExpectation::Any;
    }
}

static const char* ca_pattern_label(CAPlatePattern p) {
    switch (p) {
        case CAPlatePattern::Passenger:  return "CA_PASSENGER";
        case CAPlatePattern::Commercial1: return "CA_COMMERCIAL_1";
        case CAPlatePattern::Commercial2: return "CA_COMMERCIAL_2";
        case CAPlatePattern::Motorcycle: return "CA_MOTORCYCLE";
        case CAPlatePattern::Trailer:    return "CA_TRAILER";
        case CAPlatePattern::SixChar:    return "CA_6CHAR";
        default:                         return "CA_UNKNOWN";
    }
}

static CAPatternFit ca_best_pattern_fit(const std::string& text) {
    if (text.empty()) return {};

    // Candidate patterns and their expected lengths
    struct Candidate { CAPlatePattern pat; size_t expected_len; };
    static const Candidate candidates[] = {
        { CAPlatePattern::Passenger,  7 },
        { CAPlatePattern::Commercial1, 7 },
        { CAPlatePattern::Commercial2, 7 },
        { CAPlatePattern::Motorcycle, 6 },
        { CAPlatePattern::Trailer,    6 },
        { CAPlatePattern::SixChar,    6 },
    };

    CAPatternFit best;
    for (const auto& cand : candidates) {
        if (text.size() != cand.expected_len) continue;
        int fit = 0;
        for (size_t i = 0; i < text.size(); ++i) {
            PlateCharExpectation ex = ca_pattern_char_class(cand.pat, i);
            unsigned char c = static_cast<unsigned char>(text[i]);
            if (ex == PlateCharExpectation::Digit && std::isdigit(c)) fit++;
            else if (ex == PlateCharExpectation::Letter && std::isalpha(c)) fit++;
        }
        if (fit > best.fit_score) {
            best.fit_score = fit;
            best.pattern   = cand.pat;
            best.name      = ca_pattern_label(cand.pat);
        }
    }
    return best;
}

static PlateCharExpectation expected_plate_char_class(size_t pos, size_t len) {
  if (len == 7) {
    if (pos == 0 || pos >= 4) {
      return PlateCharExpectation::Digit;
    }
    return PlateCharExpectation::Letter;
  }

  if (len == 6) {
    if (pos < 3) {
      return PlateCharExpectation::Letter;
    }
    return PlateCharExpectation::Digit;
  }

  return PlateCharExpectation::Any;
}

static int plate_pattern_score(const std::string& text) {
  int score = 0;

  for (size_t pos = 0; pos < text.size(); ++pos) {
    PlateCharExpectation expectation = expected_plate_char_class(pos, text.size());
    if (expectation == PlateCharExpectation::Digit) {
      if (std::isdigit(static_cast<unsigned char>(text[pos]))) {
        score += 2;
      } else if (std::isalpha(static_cast<unsigned char>(text[pos]))) {
        score -= 2;
      }
    } else if (expectation == PlateCharExpectation::Letter) {
      if (std::isalpha(static_cast<unsigned char>(text[pos]))) {
        score += 2;
      } else if (std::isdigit(static_cast<unsigned char>(text[pos]))) {
        score -= 2;
      }
    }
  }

  return score;
}

static std::string choose_better_plate(const std::string& a, int a_votes,
                                       const std::string& b, int b_votes) {
    // Prefer higher votes first
    if (a_votes != b_votes) {
        return (a_votes > b_votes) ? a : b;
    }

  int a_pattern = plate_pattern_score(a);
  int b_pattern = plate_pattern_score(b);
  if (a_pattern != b_pattern) {
    return (a_pattern > b_pattern) ? a : b;
  }

    // Prefer longer plate if votes tie
    if (a.length() != b.length()) {
        return (a.length() > b.length()) ? a : b;
    }

    // Stable fallback
    return (a < b) ? a : b;
}

static std::string normalize_plate(std::string p) {
    for (auto& c : p) {
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    return p;
}

// Normalize common OCR confusions
static char normalize_char(char c) {
  return static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
}

// Add character votes from one plate read
static void add_char_votes(PlateTrack& track, const std::string& plate, int weight) {
    for (size_t i = 0; i < plate.size(); i++) {
        char c = normalize_char(plate[i]);
        track.char_votes[(int)i][c] += weight;
    }
}

// Forward declarations for pattern helpers
static bool is_letter_char(char c);
static bool is_number_char(char c);

static bool is_confusable_pair(char a, char b) {
  a = std::toupper(static_cast<unsigned char>(a));
  b = std::toupper(static_cast<unsigned char>(b));

  return
    (a == '0' && b == 'O') || (a == 'O' && b == '0') ||
    (a == '1' && b == 'I') || (a == 'I' && b == '1') ||
    (a == '7' && b == 'Z') || (a == 'Z' && b == '7') ||
    (a == '8' && b == 'B') || (a == 'B' && b == '8') ||
    (a == '2' && b == 'Z') || (a == 'Z' && b == '2') ||
    (a == '5' && b == 'S') || (a == 'S' && b == '5');
}

static int expected_class_score(char candidate, size_t pos, size_t len) {
  PlateCharExpectation generic_exp = expected_plate_char_class(pos, len);

  // Build a CA-aware expectation: if the CA pattern and generic pattern agree,
  // the signal is stronger (return 3 instead of 2).
  // Use a placeholder string filled with the candidate at pos to probe the CA fit.
  // We only check a single position here so we use ca_pattern_char_class directly.
  // Try to determine best CA pattern for this length.
  PlateCharExpectation ca_exp = PlateCharExpectation::Any;
  // For the most common CA lengths check both Passenger/SixChar expectations
  if (len == 7) {
      // Try Passenger first (most common)
      ca_exp = ca_pattern_char_class(CAPlatePattern::Passenger, pos);
  } else if (len == 6) {
      ca_exp = ca_pattern_char_class(CAPlatePattern::SixChar, pos);
  }

  // If both generic and CA agree on expectation, give bonus
  bool ca_agrees = (ca_exp != PlateCharExpectation::Any && ca_exp == generic_exp);

  if (generic_exp == PlateCharExpectation::Digit) {
    if (is_number_char(candidate)) return ca_agrees ? 3 : 2;
    if (is_letter_char(candidate)) return -1;
  } else if (generic_exp == PlateCharExpectation::Letter) {
    if (is_letter_char(candidate)) return ca_agrees ? 3 : 2;
    if (is_number_char(candidate)) return -1;
  }
  return 0;
}

  static bool same_plate_family_strict(const std::string& a, const std::string& b) {
    if (a.empty() || b.empty()) return false;
    if (std::abs((int)a.size() - (int)b.size()) > 1) return false;

    int mismatches = 0;
    size_t len = std::min(a.size(), b.size());

    for (size_t i = 0; i < len; i++) {
      if (a[i] == b[i]) continue;
      if (is_confusable_pair(a[i], b[i])) continue;

      mismatches++;
      if (mismatches > 1) return false;
    }

    mismatches += std::abs((int)a.size() - (int)b.size());
    return mismatches <= 1;
}

static int plate_event_rank(const std::string& event_type) {
  if (event_type == "LOCKED") {
    return 2;
  }
  if (event_type == "CONFIRMED") {
    return 1;
  }
  return 0;
}

static bool same_recent_plate_event_family(const std::string& left,
                                           const std::string& right) {
  std::string normalized_left = normalize_plate(left);
  std::string normalized_right = normalize_plate(right);
  if (normalized_left.empty() || normalized_right.empty()) {
    return false;
  }
  return normalized_left == normalized_right ||
         same_plate_family_strict(normalized_left, normalized_right);
}

static bool should_emit_source_plate_event(const std::string& video_source,
                                           const std::string& event_type,
                                           const std::string& plate,
                                           int confidence,
                                           int current_frame) {
  recent_plate_events.erase(
      std::remove_if(recent_plate_events.begin(), recent_plate_events.end(),
                     [current_frame](const RecentPlateEvent& event) {
                       return current_frame - event.last_seen_frame >
                              SOURCE_PLATE_EVENT_SUPPRESS_FRAMES;
                     }),
      recent_plate_events.end());

  int current_rank = plate_event_rank(event_type);
  for (RecentPlateEvent& event : recent_plate_events) {
    if (event.video_source != video_source) {
      continue;
    }
    if (!same_recent_plate_event_family(event.plate, plate)) {
      continue;
    }

    event.last_seen_frame = current_frame;
    int existing_rank = plate_event_rank(event.event_type);
    bool status_upgrade = current_rank > existing_rank;
    bool material_improvement =
      confidence >= event.confidence + SOURCE_PLATE_EVENT_IMPROVEMENT_DELTA;
    bool reemit_cooldown_elapsed =
      current_frame - event.last_emit_frame >= SOURCE_PLATE_EVENT_MIN_REEMIT_FRAMES;

    if (status_upgrade || (material_improvement && reemit_cooldown_elapsed)) {
      event.plate = plate;
      event.event_type = event_type;
      event.confidence = std::max(event.confidence, confidence);
      event.last_emit_frame = current_frame;
      return true;
    }

    if (confidence > event.confidence) {
      event.plate = plate;
      event.confidence = confidence;
    }
    if (current_rank > existing_rank) {
      event.event_type = event_type;
    }

    if (verbose_frame_logs_enabled()) {
      std::cout << "SUPPRESSED duplicate " << event_type
                << " plate=" << plate
                << " source=" << video_source
                << " existing_plate=" << event.plate
                << " confidence=" << confidence
                << " existing_confidence=" << event.confidence
                << " age_frames=" << (current_frame - event.first_frame)
                << std::endl;
    }
    return false;
  }

  recent_plate_events.push_back(RecentPlateEvent{
      video_source,
      plate,
      event_type,
      confidence,
      current_frame,
      current_frame,
      current_frame});
  return true;
}

static float rect_area(const NvOSD_RectParams& rect) {
  float width = std::max(0.0f, rect.width);
  float height = std::max(0.0f, rect.height);
  return width * height;
}

static float rect_intersection_area(const NvOSD_RectParams& a,
                                    const NvOSD_RectParams& b) {
  float left = std::max(a.left, b.left);
  float top = std::max(a.top, b.top);
  float right = std::min(a.left + a.width, b.left + b.width);
  float bottom = std::min(a.top + a.height, b.top + b.height);
  float width = std::max(0.0f, right - left);
  float height = std::max(0.0f, bottom - top);
  return width * height;
}

static float rect_iou(const NvOSD_RectParams& a, const NvOSD_RectParams& b) {
  float intersection = rect_intersection_area(a, b);
  if (intersection <= 0.0f) {
    return 0.0f;
  }

  float union_area = rect_area(a) + rect_area(b) - intersection;
  if (union_area <= 0.0f) {
    return 0.0f;
  }

  return intersection / union_area;
}

static bool rect_contains_center(const NvOSD_RectParams& outer,
                                 const NvOSD_RectParams& inner) {
  float center_x = inner.left + (inner.width * 0.5f);
  float center_y = inner.top + (inner.height * 0.5f);
  return center_x >= outer.left &&
         center_x <= outer.left + outer.width &&
         center_y >= outer.top &&
         center_y <= outer.top + outer.height;
}

static bool extract_track_id(const NvDsObjectMeta* vehicle_meta, uint64_t& track_id) {
  track_id = 0;
  if (!vehicle_meta) {
    return false;
  }

  uint64_t object_id = vehicle_meta->object_id;
  if (object_id == std::numeric_limits<uint64_t>::max() ||
      object_id == UNTRACKED_OBJECT_ID) {
    return false;
  }

  track_id = object_id;
  return true;
}

static NvDsObjectMeta* resolve_vehicle_meta_for_plate(NvDsFrameMeta* frame_meta,
                                                      NvDsObjectMeta* plate_meta) {
  if (!plate_meta) {
    return NULL;
  }

  NvDsObjectMeta* parent_meta = plate_meta->parent;
  uint64_t track_id = 0;
  if (extract_track_id(parent_meta, track_id)) {
    return parent_meta;
  }

  NvDsObjectMeta* best_candidate = NULL;
  float best_score = -1.0f;

  for (NvDsMetaList* candidate_list = frame_meta ? frame_meta->obj_meta_list : NULL;
       candidate_list != NULL;
       candidate_list = candidate_list->next) {
    NvDsObjectMeta* candidate = static_cast<NvDsObjectMeta*>(candidate_list->data);
    if (!candidate) {
      continue;
    }

    if (candidate->unique_component_id != PRIMARY_DETECTOR_UID ||
        candidate->class_id != PGIE_CLASS_ID_VEHICLE) {
      continue;
    }

    if (!rect_contains_center(candidate->rect_params, plate_meta->rect_params)) {
      continue;
    }

    float score = rect_iou(candidate->rect_params, plate_meta->rect_params);
    if (parent_meta) {
      score += rect_iou(candidate->rect_params, parent_meta->rect_params) * 4.0f;
    }

    if (candidate == parent_meta) {
      score += 10.0f;
    }

    uint64_t candidate_track_id = 0;
    if (extract_track_id(candidate, candidate_track_id)) {
      score += 2.0f;
    }

    if (score > best_score) {
      best_score = score;
      best_candidate = candidate;
    }
  }

  if (best_candidate) {
    return best_candidate;
  }

  return parent_meta ? parent_meta : plate_meta;
}

  static PlateConfidenceResult compute_plate_confidence(
    const PlateTrack& track,
    const std::string& consensus_plate,
    int best_votes,
    int second_votes)
  {
    PlateConfidenceResult r;

    r.vote_points = std::min(20, best_votes / 2);
    r.stability_points = std::min(15, track.stable_frames * 1);

    if (track.stable_frames >= 3 && best_votes >= 3) {
      r.bonus_points += 8;
    }

    if (track.stable_frames >= 5) {
      r.bonus_points += 5;
    }

    int dominance_sum = 0;
    int counted_positions = 0;

    for (size_t pos = 0; pos < consensus_plate.size(); pos++) {
      auto it = track.char_votes.find((int)pos);
      if (it == track.char_votes.end() || it->second.empty()) continue;

      int best = 0;
      int second = 0;

      for (const auto& kv : it->second) {
        int v = kv.second;
        if (v > best) {
          second = best;
          best = v;
        } else if (v > second) {
          second = v;
        }
      }

      int gap = best - second;
      int pos_score = 0;
      if (gap >= 5) pos_score = 4;
      else if (gap >= 4) pos_score = 3;
      else if (gap >= 3) pos_score = 2;
      else if (gap >= 2) pos_score = 1;
      else pos_score = -1;

      dominance_sum += pos_score;
      counted_positions++;

      if (pos_score < 0) r.penalty_points += pos_score * 2;
    }

    if (counted_positions > 0) {
      r.dominance_points = std::min(30, (dominance_sum * 30) / (counted_positions * 4));
    }

    int sep = best_votes - second_votes;
    if (second_votes == 0) r.separation_points = 10;
    else if (sep >= 10) r.separation_points = 10;
    else if (sep >= 7) r.separation_points = 8;
    else if (sep >= 5) r.separation_points = 6;
    else if (sep >= 3) r.separation_points = 3;
    else r.separation_points = 0;

    if (consensus_plate.find('?') != std::string::npos) {
      r.penalty_points -= 20;
    }

    if (consensus_plate.size() < 5 || consensus_plate.size() > 8) {
      r.penalty_points -= 10;
    }

    if (second_votes > 0 && sep <= 2) {
      r.penalty_points -= 10;
    }

    int confusable_penalty = 0;

    for (size_t pos = 0; pos < consensus_plate.size(); pos++) {
      auto it = track.char_votes.find((int)pos);
      if (it == track.char_votes.end()) continue;

      int best = 0;
      int second = 0;
      char best_char = '?';
      char second_char = '?';

      for (const auto& kv : it->second) {
        if (kv.second > best) {
          second = best;
          second_char = best_char;
          best = kv.second;
          best_char = kv.first;
        } else if (kv.second > second) {
          second = kv.second;
          second_char = kv.first;
        }
      }

      if (is_confusable_pair(best_char, second_char)) {
        confusable_penalty += 5;
      }
    }

    r.penalty_points -= confusable_penalty;

    int inconsistent_positions = 0;

    for (const auto& kv : track.char_votes) {
      if (kv.second.size() >= 3) {
        inconsistent_positions++;
      }
    }

    r.penalty_points -= inconsistent_positions * 2;

    if (second_votes > 0) {
      double ratio = static_cast<double>(second_votes) / best_votes;
      if (ratio > 0.7) {
        r.penalty_points -= 15;
      } else if (ratio > 0.5) {
        r.penalty_points -= 8;
      }
    }

        r.score = r.vote_points + r.stability_points + r.dominance_points +
          r.bonus_points +
          r.separation_points + r.penalty_points;

    // CA pattern bonus/penalty
    {
        CAPatternFit ca = ca_best_pattern_fit(consensus_plate);
        r.ca_pattern_name = ca.name;
        size_t len = consensus_plate.size();
        if (len > 0) {
            // fit_score out of len -> fraction [0..1]
            double frac = static_cast<double>(ca.fit_score) / static_cast<double>(len);
            if (frac >= 1.0) r.ca_pattern_points = 8;
            else if (frac >= 0.857) r.ca_pattern_points = 5;  // 6/7
            else if (frac >= 0.714) r.ca_pattern_points = 2;  // 5/7
            else if (frac >= 0.5)   r.ca_pattern_points = -2;
            else                    r.ca_pattern_points = -5;
        }
        r.score += r.ca_pattern_points;
    }

    if (r.score < 0) r.score = 0;
    if (r.score > 100) r.score = 100;

    return r;
  }

static std::string build_consensus_plate_generic(const PlateTrack& track) {
    if (track.char_votes.empty()) return "";

  std::map<int, std::map<char, int>> boosted_char_votes = track.char_votes;
  std::map<std::string, int> full_plate_votes = track.votes;

  for (const auto& kv : full_plate_votes) {
    const std::string& plate = kv.first;
    int votes = kv.second;

    if (votes < 3) {
      continue;
    }

    for (size_t i = 0; i < plate.size(); i++) {
      boosted_char_votes[(int)i][normalize_char(plate[i])] += 2;
    }
  }

    int max_pos = -1;
  for (const auto& kv : boosted_char_votes) {
        if (kv.first > max_pos) max_pos = kv.first;
    }

  auto it0 = boosted_char_votes.find(0);
  if (it0 != boosted_char_votes.end()) {
    int total = 0;
    int best = 0;

    for (const auto& kv : it0->second) {
      total += kv.second;
      best = std::max(best, kv.second);
    }

    if (total > 0 && best < total * 0.6) {
      return "";
    }
  }

    std::string result;

    for (int pos = 0; pos <= max_pos; pos++) {
    auto it = boosted_char_votes.find(pos);
    if (it == boosted_char_votes.end() || it->second.empty()) {
            result.push_back('?');
            continue;
        }

    int total_letter_votes = 0;
    int total_digit_votes = 0;

    for (const auto& kv : it->second) {
      char c = kv.first;
      int votes = kv.second;

      if (is_letter_char(c)) total_letter_votes += votes;
      if (is_number_char(c)) total_digit_votes += votes;
    }

    bool prefer_letter = total_letter_votes > total_digit_votes;
    bool prefer_digit = total_digit_votes > total_letter_votes;

        char best_char = '?';
    int best_score = -1;
    int second_score = -1;

    for (const auto& kv : it->second) {
      char candidate = kv.first;
      int score = kv.second;

      if (prefer_letter && is_letter_char(candidate)) score += 1;
      if (prefer_digit && is_number_char(candidate)) score += 1;
      score += expected_class_score(candidate, static_cast<size_t>(pos), static_cast<size_t>(max_pos + 1));

            if (score > best_score) {
                second_score = best_score;
                best_score = score;
        best_char = candidate;
            } else if (score > second_score) {
                second_score = score;
            }
        }

    if (second_score >= 0 && (best_score - second_score) <= 1) {
      char adjusted_best = best_char;
      int adjusted_best_score = best_score;

      for (const auto& kv : it->second) {
        char candidate = kv.first;
        int score = kv.second;

        if (prefer_letter && is_letter_char(candidate)) score += 1;
        if (prefer_digit && is_number_char(candidate)) score += 1;
        score += expected_class_score(candidate, static_cast<size_t>(pos), static_cast<size_t>(max_pos + 1));

        if (is_confusable_pair(candidate, best_char)) score += 1;

        if (score > adjusted_best_score) {
          adjusted_best_score = score;
          adjusted_best = candidate;
        }
      }

      best_char = adjusted_best;
      best_score = adjusted_best_score;
        }

    if (best_score < 3) {
      result.push_back('?');
      continue;
    }

    if (second_score >= 0 && best_score < second_score * 1.5) {
      result.push_back('?');
      continue;
    }

    if (second_score >= 0 && (best_score - second_score) <= 0) {
      result.push_back('?');
        } else {
            result.push_back(best_char);
        }
    }

    return result;
}

  static char resolve_confusable_char(
    const std::map<char, int>& pos_votes,
    char current_char,
    size_t pos,
    size_t len)
  {
    if (pos_votes.empty()) return current_char;

    int digit_votes = 0;
    int letter_votes = 0;

    for (const auto& kv : pos_votes) {
      char c = kv.first;
      int v = kv.second;

      if (std::isdigit(static_cast<unsigned char>(c))) digit_votes += v;
      if (std::isalpha(static_cast<unsigned char>(c))) letter_votes += v;
    }

    char best_alt = current_char;
    int best_alt_votes = -1;

    for (const auto& kv : pos_votes) {
      char candidate = kv.first;
      int votes = kv.second;

      if (!is_confusable_pair(candidate, current_char) && candidate != current_char)
        continue;

      int score = votes;

      if (digit_votes > letter_votes &&
        std::isdigit(static_cast<unsigned char>(candidate))) {
        score += 1;
      }

      if (letter_votes > digit_votes &&
        std::isalpha(static_cast<unsigned char>(candidate))) {
        score += 1;
      }

      score += expected_class_score(candidate, pos, len);

      if (score > best_alt_votes) {
        best_alt_votes = score;
        best_alt = candidate;
      }
    }

    return best_alt;
  }

  static std::string refine_consensus_with_confusables(
    const PlateTrack& track,
    const std::string& consensus)
  {
    std::string refined = consensus;

    for (size_t pos = 0; pos < refined.size(); pos++) {
      auto it = track.char_votes.find((int)pos);
      if (it == track.char_votes.end()) continue;

      refined[pos] = resolve_confusable_char(it->second, refined[pos], pos, refined.size());
    }

    return refined;
  }

static char paired_confusable_char(char c) {
  switch (std::toupper(static_cast<unsigned char>(c))) {
    case '0': return 'O';
    case 'O': return '0';
    case '1': return 'I';
    case 'I': return '1';
    case '7': return 'Z';
    case 'Z': return '7';
    case '8': return 'B';
    case 'B': return '8';
    case '5': return 'S';
    case 'S': return '5';
    default: return '\0';
  }
}

static char finalize_confusable_char(const std::map<char, int>& pos_votes,
                                     char current_char,
                                     size_t pos,
                                     size_t len) {
  char alternate = paired_confusable_char(current_char);
  if (alternate == '\0') {
    return current_char;
  }

  int current_votes = 0;
  int alternate_votes = 0;
  int digit_votes = 0;
  int letter_votes = 0;

  for (const auto& kv : pos_votes) {
    char candidate = std::toupper(static_cast<unsigned char>(kv.first));
    int votes = kv.second;
    if (candidate == current_char) {
      current_votes += votes;
    }
    if (candidate == alternate) {
      alternate_votes += votes;
    }
    if (std::isdigit(static_cast<unsigned char>(candidate))) {
      digit_votes += votes;
    }
    if (std::isalpha(static_cast<unsigned char>(candidate))) {
      letter_votes += votes;
    }
  }

  if (current_votes == 0 && alternate_votes == 0) {
    return current_char;
  }

  int current_score = (current_votes * 2) + expected_class_score(current_char, pos, len);
  int alternate_score = (alternate_votes * 2) + expected_class_score(alternate, pos, len);

  if (digit_votes > letter_votes) {
    if (std::isdigit(static_cast<unsigned char>(current_char))) current_score += 1;
    if (std::isdigit(static_cast<unsigned char>(alternate))) alternate_score += 1;
  } else if (letter_votes > digit_votes) {
    if (std::isalpha(static_cast<unsigned char>(current_char))) current_score += 1;
    if (std::isalpha(static_cast<unsigned char>(alternate))) alternate_score += 1;
  }

  if (alternate_score > current_score) {
    return alternate;
  }
  return current_char;
}

static std::string finalize_consensus_plate(const PlateTrack& track, const std::string& raw) {
    std::string out = raw;
    for (size_t i = 0; i < out.size(); i++) {
        auto it = track.char_votes.find((int)i);
        if (it == track.char_votes.end()) {
            continue;
        }
        out[i] = finalize_confusable_char(it->second, out[i], i, out.size());
    }
    return out;
}

static bool is_letter_char(char c) {
    return std::isalpha(static_cast<unsigned char>(c));
}

static bool is_number_char(char c) {
    return std::isdigit(static_cast<unsigned char>(c));
}

extern "C" void parse_nvdsanalytics_meta_data(NvDsBatchMeta *batch_meta);

typedef struct _perf_measure {
  GstClockTime pre_time;
  GstClockTime total_time;
  guint count;
} perf_measure;

/* ── Large-vehicle license-plate zone probes ──────────────────────────────── *
 * Problem: for tall/wide vehicles (box trucks, tractor-trailers, large vans), *
 * the LPDNet secondary GIE receives the entire vehicle bounding box cropped   *
 * and scaled to its 480×640 input.  The plate region shrinks to just a few    *
 * pixels and becomes undetectable.                                             *
 *                                                                              *
 * Solution: a pre-LPD probe shrinks large-vehicle bounding boxes to their     *
 * bottom plate zone before the secondary GIE crops them.  A post-LPR probe   *
 * restores the original bounding boxes so attribute classifiers (make, type,  *
 * color) still receive the full vehicle crop.                                  *
 *                                                                              *
 * Tuning knobs (frame is 960×544):                                             */
#define LV_HEIGHT_THRESH 160.0f  /* px – vehicles taller than this are "large" */
#define LV_WIDTH_THRESH  380.0f  /* px – OR wider than this are "large"         */
#define LV_PLATE_FRAC      0.40f /* keep bottom 40 % of the box for LPD         */
#define LV_PLATE_MIN_H    80.0f  /* never shrink the crop below this height      */

/* misc_obj_info indices used to stash the original bounding box between probes.
 * MAX_USER_FIELDS == 4, so valid indices are 0–3. */
#define LV_MAGIC_IDX    0
#define LV_ORIG_TOP_IDX 1
#define LV_ORIG_H_IDX   2
#define LV_MAGIC_VAL    ((gint64)0x4C564144LL)  /* "LVAD" = Large-Vehicle ADjusted */

/* Pre-LPD probe: runs on queue4 src pad (between tracker and secondary_detector).
 * For large vehicles, adjusts rect_params to the bottom plate zone. */
static GstPadProbeReturn
pre_lpd_large_vehicle_crop_probe(GstPad * /*pad*/, GstPadProbeInfo *info, gpointer /*user_data*/) {
  GstBuffer *buf = GST_PAD_PROBE_INFO_BUFFER(info);
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
  if (!batch_meta) return GST_PAD_PROBE_OK;

  for (NvDsMetaList *lf = batch_meta->frame_meta_list; lf; lf = lf->next) {
    NvDsFrameMeta *fm = (NvDsFrameMeta *)lf->data;
    for (NvDsMetaList *lo = fm->obj_meta_list; lo; lo = lo->next) {
      NvDsObjectMeta *obj = (NvDsObjectMeta *)lo->data;
      if (!obj) continue;
      if (obj->unique_component_id != PRIMARY_DETECTOR_UID) continue;
      if (obj->class_id != PGIE_CLASS_ID_VEHICLE) continue;

      float h = obj->rect_params.height;
      float w = obj->rect_params.width;

      /* Only adjust bounding boxes that are large enough to hide the plate */
      if (h < LV_HEIGHT_THRESH && w < LV_WIDTH_THRESH) continue;

      /* Plate zone: bottom LV_PLATE_FRAC of the box, at least LV_PLATE_MIN_H */
      float plate_h = std::max(h * LV_PLATE_FRAC, LV_PLATE_MIN_H);
      if (plate_h >= h) continue;  /* already small enough — no change */

      float new_top = obj->rect_params.top + (h - plate_h);

      /* Stash originals so the restore probe can undo this adjustment */
      obj->misc_obj_info[LV_MAGIC_IDX]    = LV_MAGIC_VAL;
      obj->misc_obj_info[LV_ORIG_TOP_IDX] = (gint64)obj->rect_params.top;
      obj->misc_obj_info[LV_ORIG_H_IDX]   = (gint64)obj->rect_params.height;

      obj->rect_params.top    = new_top;
      obj->rect_params.height = plate_h;
    }
  }
  return GST_PAD_PROBE_OK;
}

/* Post-LPR restore probe: runs on queue6 src pad (after LPR, before attribute
 * classifiers).  Restores the original full-vehicle bounding boxes so that
 * make/type/color classifiers get the correct vehicle crop. */
static GstPadProbeReturn
post_lpr_restore_bbox_probe(GstPad * /*pad*/, GstPadProbeInfo *info, gpointer /*user_data*/) {
  GstBuffer *buf = GST_PAD_PROBE_INFO_BUFFER(info);
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
  if (!batch_meta) return GST_PAD_PROBE_OK;

  for (NvDsMetaList *lf = batch_meta->frame_meta_list; lf; lf = lf->next) {
    NvDsFrameMeta *fm = (NvDsFrameMeta *)lf->data;
    for (NvDsMetaList *lo = fm->obj_meta_list; lo; lo = lo->next) {
      NvDsObjectMeta *obj = (NvDsObjectMeta *)lo->data;
      if (!obj) continue;
      if (obj->misc_obj_info[LV_MAGIC_IDX] != LV_MAGIC_VAL) continue;

      /* Restore the original bounding box */
      obj->rect_params.top    = (gfloat)obj->misc_obj_info[LV_ORIG_TOP_IDX];
      obj->rect_params.height = (gfloat)obj->misc_obj_info[LV_ORIG_H_IDX];
      obj->misc_obj_info[LV_MAGIC_IDX] = 0;  /* clear sentinel */
    }
  }
  return GST_PAD_PROBE_OK;
}

/* osd_sink_pad_buffer_probe  will extract metadata received on OSD sink pad
 * and update params for drawing rectangle, object information etc. */
static GstPadProbeReturn
osd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data) {
  GstBuffer *buf = (GstBuffer *)info->data;
  NvDsObjectMeta *obj_meta = NULL;
  guint vehicle_count = 0;
  guint person_count = 0;
  guint lp_count = 0;
  guint label_i = 0;
  NvDsMetaList *l_frame = NULL;
  NvDsMetaList *l_obj = NULL;
  NvDsMetaList *l_class = NULL;
  NvDsMetaList *l_label = NULL;
  NvDsDisplayMeta *display_meta = NULL;
  NvDsClassifierMeta *class_meta = NULL;
  NvDsLabelInfo *label_info = NULL;
  GstClockTime now;
  perf_measure *perf = (perf_measure *)(u_data);

  if (!perf) {
    return GST_PAD_PROBE_OK;
  }

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

  now = g_get_monotonic_time();

  if (perf->pre_time == GST_CLOCK_TIME_NONE) {
    perf->pre_time = now;
    perf->total_time = GST_CLOCK_TIME_NONE;
  } else {
    if (perf->total_time == GST_CLOCK_TIME_NONE) {
      perf->total_time = (now - perf->pre_time);
    } else {
      perf->total_time += (now - perf->pre_time);
    }
    perf->pre_time = now;
    perf->count++;
  }

  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
       l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
    int offset = 0;
    if (!frame_meta)
      continue;

    std::string video_source = build_video_source_name(frame_meta);
    int runtime_frame_number = frame_meta->frame_num;
    if (runtime_frame_number <= 0 && frame_number > 0) {
      runtime_frame_number = frame_number;
    }
    update_runtime_source_status(current_case_id(), video_source, runtime_frame_number);
    std::vector<RuntimePreviewDetection> frame_preview_detections;
    bool preview_due = should_update_runtime_source_preview(video_source);
    cv::Mat frame_mat;
    bool frame_mat_loaded = false;

    auto get_frame_mat = [&]() -> const cv::Mat& {
      if (!frame_mat_loaded) {
        frame_mat = extract_frame_mat(buf, frame_meta->batch_id);
        frame_mat_loaded = true;
      }
      return frame_mat;
    };

    for (auto& track_entry : plate_tracks) {
      decay_plate_track(track_entry.second, frame_number);
    }

    for (auto it = plate_tracks.begin(); it != plate_tracks.end();) {
      int unseen = frame_number - it->second.last_seen_frame;
      if (unseen > TRACK_DROP_FRAMES) {
        it = plate_tracks.erase(it);
      } else {
        ++it;
      }
    }

    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
         l_obj = l_obj->next) {
      obj_meta = (NvDsObjectMeta *)(l_obj->data);

      if (!obj_meta)
        continue;

      RuntimePreviewDetection *preview_detection = NULL;

      /* Check that the object has been detected by the primary detector
       * and that the class id is that of vehicles/persons. */
      if (obj_meta->unique_component_id == PRIMARY_DETECTOR_UID) {
        if (obj_meta->class_id == PGIE_CLASS_ID_VEHICLE)
          vehicle_count++;
        if (obj_meta->class_id == PGIE_CLASS_ID_PERSON)
          person_count++;
      }

      if (obj_meta->unique_component_id == SECONDARY_DETECTOR_UID) {
        if (obj_meta->class_id == SGIE_CLASS_ID_LPD) {
          lp_count++;
          frame_preview_detections.push_back(RuntimePreviewDetection{
              static_cast<int>(obj_meta->rect_params.left),
              static_cast<int>(obj_meta->rect_params.top),
              static_cast<int>(obj_meta->rect_params.width),
              static_cast<int>(obj_meta->rect_params.height),
              0,
              std::string(),
              "out_of_focus"});
          preview_detection = &frame_preview_detections.back();
          /* Print this info only when operating in secondary model. */
          if (verbose_frame_logs_enabled() && obj_meta->parent)
            g_print("License plate found for parent object %p (type=%s)\n",
                    obj_meta->parent,
                    pgie_classes_str[obj_meta->parent->class_id]);

          obj_meta->text_params.set_bg_clr = 1;
          obj_meta->text_params.text_bg_clr.red = 0.0;
          obj_meta->text_params.text_bg_clr.green = 0.0;
          obj_meta->text_params.text_bg_clr.blue = 0.0;
          obj_meta->text_params.text_bg_clr.alpha = 0.0;

          obj_meta->text_params.font_params.font_color.red = 1.0;
          obj_meta->text_params.font_params.font_color.green = 1.0;
          obj_meta->text_params.font_params.font_color.blue = 0.0;
          obj_meta->text_params.font_params.font_color.alpha = 1.0;
          obj_meta->text_params.font_params.font_size = 12;
        }
      }

      for (l_class = obj_meta->classifier_meta_list; l_class != NULL;
           l_class = l_class->next) {
        class_meta = (NvDsClassifierMeta *)(l_class->data);
        if (!class_meta)
          continue;
        if (class_meta->unique_component_id == SECONDARY_CLASSIFIER_UID) {
          for (label_i = 0, l_label = class_meta->label_info_list;
               label_i < class_meta->num_labels && l_label;
               label_i++, l_label = l_label->next) {
            label_info = (NvDsLabelInfo *)(l_label->data);
            if (label_info) {
              if (label_info->label_id == 0 &&
                  label_info->result_class_id == 1) {
                std::string plate_str = get_plate_string(label_info);
                if (preview_detection && !plate_str.empty()) {
                  preview_detection->plate = plate_str;
                }
                if (plate_str.length() < 4) continue;
                if (!is_valid_plate_text(plate_str))
                  continue;

                NvDsObjectMeta *vehicle_meta = resolve_vehicle_meta_for_plate(frame_meta, obj_meta);
                uint64_t track_id = 0;
                bool track_id_valid = extract_track_id(vehicle_meta, track_id);
                if (!track_id_valid) {
                  track_id_valid = extract_track_id(obj_meta, track_id);
                }
                std::string plate_track_key =
                  build_plate_track_key(video_source, track_id_valid, track_id, obj_meta);

                std::pair<std::map<std::string, PlateTrack>::iterator, bool> track_insert =
                  plate_tracks.emplace(plate_track_key, PlateTrack());
                auto& track = track_insert.first->second;
                if (track_insert.second || track.first_seen_frame < 0) {
                  track.first_seen_frame = frame_number;
                  track.first_seen_time_us = now;
                }
                VehicleAttributeObservations attribute_observations =
                  extract_vehicle_attribute_observations(vehicle_meta);

                int weight = plate_vote_weight(plate_str, label_info->result_prob);
                int expected_len = 6; // or 7 depending on CA plates
                if (abs((int)plate_str.length() - expected_len) > 1)
                    weight -= 2;
                bool merged = false;

                // Try to merge this plate into an existing family
                for (auto it = track.votes.begin(); it != track.votes.end(); ++it) {
                    if (same_plate_family(normalize_plate(it->first), normalize_plate(plate_str))) {
                        std::string old_key = it->first;
                        int old_votes = it->second;

                        // Add weight to the family
                        int new_votes = old_votes + weight;

                        // Decide whether old_key or plate_str should be the canonical key
                        std::string best_key = choose_better_plate(old_key, old_votes, plate_str, weight);

                        if (best_key == old_key) {
                            track.votes[old_key] = new_votes;
                        } else {
                            track.votes.erase(old_key);
                            track.votes[best_key] = new_votes;
                        }

                        merged = true;
                        break;
                    }
                }

                    add_char_votes(track, plate_str, weight);

                if (!merged) {
                    track.votes[plate_str] += weight;
                }
                track.last_seen_frame = frame_number;
                track.missed_frames = 0;

                std::string best_plate;
                int best_votes = 0;
                std::string second_plate;
                int second_votes = 0;

                for (auto& v : track.votes) {
                    if (v.second > best_votes) {
                    second_plate = best_plate;
                    second_votes = best_votes;
                        best_plate = v.first;
                        best_votes = v.second;
                  } else if (v.second > second_votes) {
                    second_plate = v.first;
                    second_votes = v.second;
                    }
                }

                if (!best_plate.empty() && !second_plate.empty()) {
                  if (same_plate_family_strict(best_plate, second_plate)) {
                    if (best_votes >= second_votes + 3) {
                      second_plate.clear();
                      second_votes = 0;
                    }
                  }
                }

                std::string consensus_plate =
                  finalize_consensus_plate(track, build_consensus_plate_generic(track));

                consensus_plate =
                  finalize_consensus_plate(track, refine_consensus_with_confusables(track, consensus_plate));

                if (preview_detection && !consensus_plate.empty()) {
                  preview_detection->plate = consensus_plate;
                }

                if (consensus_plate.length() < 5) {
                  continue;
                }

                PlateConfidenceResult conf =
                  compute_plate_confidence(track, consensus_plate, best_votes, second_votes);
                VehicleAttributes vehicle_attributes = resolve_vehicle_attributes(track);

                bool preview_plate_readable =
                  is_valid_plate_text(consensus_plate) &&
                  consensus_plate.find('?') == std::string::npos;
                if (preview_detection) {
                  preview_detection->confidence = conf.score;
                  preview_detection->focus_state =
                    preview_focus_state_for_plate(conf.score, preview_plate_readable);
                }

                if (conf.score < 40) {
                  continue;
                }

                if (track.first_readable_frame < 0) {
                  track.first_readable_frame = frame_number;
                  track.first_readable_time_us = now;
                }

                if (verbose_frame_logs_enabled()) {
                  std::cout << "TRACK UPDATE: " << consensus_plate
                            << " score=" << conf.score
                            << " reported=" << track.reported
                            << std::endl;
                }

                if (consensus_plate == track.last_best) {
                    track.stable_frames++;
                } else if (same_plate_family_strict(consensus_plate, track.last_best)) {
                    // Within 1-char noise tolerance — count as stable
                    track.stable_frames++;
                    track.last_best = consensus_plate;
                } else {
                    track.stable_frames = 0;
                    track.last_best = consensus_plate;
                }

                VehicleCropQuality vehicle_crop_quality =
                  assess_vehicle_crop_quality(get_frame_mat(), vehicle_meta);
                penalize_attribute_votes_for_bad_lighting(track, vehicle_crop_quality);
                remember_large_vehicle_track_hints(track, attribute_observations);

                bool likely_large_vehicle =
                  is_likely_large_vehicle_track(track, attribute_observations);

                VehicleAttributeVoteContext attribute_vote_context =
                  build_vehicle_attribute_vote_context(vehicle_meta,
                                                      consensus_plate,
                                                      conf,
                                                      best_votes,
                                                      track.stable_frames,
                                                      vehicle_crop_quality,
                                                      likely_large_vehicle);
                add_vehicle_attribute_votes(vehicle_meta,
                                            track,
                                            attribute_observations,
                                            attribute_vote_context);
                add_visual_color_vote(track,
                                      estimate_visual_vehicle_color(get_frame_mat(),
                                                                    vehicle_meta,
                                                                    obj_meta,
                                                                    vehicle_crop_quality),
                                      attribute_vote_context);
                vehicle_attributes = resolve_vehicle_attributes(track);
                apply_recent_plate_attribute_prior(consensus_plate,
                                                   video_source,
                                                   frame_number,
                                                   vehicle_crop_quality,
                                                   attribute_observations,
                                                   vehicle_attributes);
                remember_recent_plate_attributes(consensus_plate,
                                                 video_source,
                                                 frame_number,
                                                 conf.score,
                                                 track,
                                                 attribute_observations,
                                                 vehicle_attributes);

                if (should_log_large_vehicle_make_debug(consensus_plate)) {
                  bool prefer_pickup =
                    prefer_pickup_large_vehicle_make(track, attribute_observations);
                  std::cout << "TARGETED VEHICLE DEBUG: plate=" << consensus_plate
                            << " frame=" << frame_number
                            << " likely_large_vehicle="
                            << (likely_large_vehicle ? "1" : "0")
                            << " type_obs=" << attribute_observations.type.label
                            << '@' << format_probability(attribute_observations.type.probability)
                            << " make_obs=" << attribute_observations.make.label
                            << '@' << format_probability(attribute_observations.make.probability)
                            << " color_obs=" << attribute_observations.color.label
                            << '@' << format_probability(attribute_observations.color.probability)
                            << " prefer_pickup=" << (prefer_pickup ? "1" : "0")
                            << " resolved_make=" << vehicle_attributes.make
                            << " resolved_type=" << vehicle_attributes.type
                            << " resolved_color=" << vehicle_attributes.color
                            << " make_votes=[" << summarize_attribute_vote_map(track.make_votes) << ']'
                            << " type_votes=[" << summarize_attribute_vote_map(track.type_votes) << ']'
                            << " color_votes=[" << summarize_attribute_vote_map(track.color_votes) << ']'
                            << " type_candidates=["
                            << summarize_classifier_candidates(vehicle_meta,
                                                              VEHICLE_TYPE_CLASSIFIER_UID)
                            << ']'
                            << " make_candidates=["
                            << summarize_classifier_candidates(vehicle_meta,
                                                              VEHICLE_MAKE_CLASSIFIER_UID)
                            << ']'
                            << " color_candidates=["
                            << summarize_classifier_candidates(vehicle_meta,
                                                              VEHICLE_COLOR_CLASSIFIER_UID)
                            << ']';
                  if (likely_large_vehicle) {
                    std::cout << " large_vehicle_make_candidates=["
                              << summarize_large_vehicle_make_candidates(vehicle_meta,
                                                                         prefer_pickup)
                              << ']';
                  }
                  std::cout
                            << std::endl;
                }

                bool qualifies_for_debug =
                  debug_events_enabled() &&
                  best_votes >= 25 &&
                  track.stable_frames >= DEBUG_STABLE_FRAMES &&
                  is_valid_plate_text(consensus_plate) &&
                  looks_like_reasonable_us_plate(consensus_plate) &&
                  consensus_plate.find('?') == std::string::npos &&
                  conf.score >= DEBUG_THRESHOLD_MIN &&
                  should_emit_track_event(consensus_plate, conf.score,
                                          track.last_debug_plate,
                                          track.last_debug_confidence,
                                          track.last_debug_frame,
                                          frame_number) &&
                  conf.score < REJECT_THRESHOLD;

                if (qualifies_for_debug) {
                  int veh_left = vehicle_meta ? static_cast<int>(vehicle_meta->rect_params.left) : 0;
                  int veh_top = vehicle_meta ? static_cast<int>(vehicle_meta->rect_params.top) : 0;
                  int veh_width = vehicle_meta ? static_cast<int>(vehicle_meta->rect_params.width) : 0;
                  int veh_height = vehicle_meta ? static_cast<int>(vehicle_meta->rect_params.height) : 0;
                  int plate_left = static_cast<int>(obj_meta->rect_params.left);
                  int plate_top = static_cast<int>(obj_meta->rect_params.top);
                  int plate_width = static_cast<int>(obj_meta->rect_params.width);
                  int plate_height = static_cast<int>(obj_meta->rect_params.height);

                    bool debug_ok = write_debug_event(
                      get_frame_mat(),
                      g_evidence_root,
                      video_source,
                      kModelVersion,
                      consensus_plate,
                      vehicle_attributes.make,
                      vehicle_attributes.type,
                      vehicle_attributes.color,
                      conf.score,
                      frame_number,
                      track_id_valid,
                      track_id,
                      veh_left, veh_top, veh_width, veh_height,
                      plate_left, plate_top, plate_width, plate_height);

                  std::cout << "DEBUG Plate: " << consensus_plate
                            << " [confidence=" << conf.score << "]";
                  if (!debug_ok) std::cout << " [debug write failed]";
                  std::cout << std::endl;

                  track.last_debug_plate = consensus_plate;
                  track.last_debug_confidence = conf.score;
                  track.last_debug_frame = frame_number;
                }

                if (!track.reported &&
                  best_votes >= 15 &&
                  track.stable_frames >= 1 &&
                  is_valid_plate_text(consensus_plate) &&
                  looks_like_reasonable_us_plate(consensus_plate) &&
                  conf.score < REJECT_THRESHOLD) {

                  if (verbose_frame_logs_enabled()) {
                    std::cout << "REJECTED Plate: " << consensus_plate
                              << " [confidence=" << conf.score << "]"
                              << std::endl;
                  }
                }

                if (best_votes >= 15 &&
                  track.stable_frames >= 1 &&
                  is_valid_plate_text(consensus_plate) &&
                  looks_like_reasonable_us_plate(consensus_plate) &&
                  consensus_plate.find('?') == std::string::npos &&
                  conf.score >= REJECT_THRESHOLD) {

                  // CONFIRMED fires only once per track (when first reported).
                  // This creates the card on screen without churning on every
                  // 2-point confidence improvement.
                  bool should_emit_confirmed =
                    !track.reported &&
                    should_emit_track_event(consensus_plate, conf.score,
                                            track.last_confirmed_plate,
                                            track.last_confirmed_confidence,
                                            track.last_confirmed_frame,
                                            frame_number);

                  // LOCKED requires a much larger confidence delta (TRACK_LOCK_IMPROVEMENT_DELTA)
                  // so that only meaningful improvements trigger a re-lock event.
                  bool should_emit_locked =
                    should_lock_plate(consensus_plate, conf.score, track.stable_frames) &&
                    best_votes >= 35 &&
                    (track.last_locked_plate.empty() ||
                     track.last_locked_plate != consensus_plate ||
                     conf.score >= track.last_locked_confidence + TRACK_LOCK_IMPROVEMENT_DELTA ||
                     (frame_number - track.last_locked_frame) >= TRACK_EVENT_COOLDOWN_FRAMES);

                  if (should_emit_confirmed && should_emit_locked) {
                    should_emit_confirmed = false;
                  }

                  if (!should_emit_confirmed && !should_emit_locked) {
                    continue;
                  }

                  if (should_emit_confirmed &&
                      !should_emit_source_plate_event(video_source,
                                                      "CONFIRMED",
                                                      consensus_plate,
                                                      conf.score,
                                                      frame_number)) {
                    track.reported = true;
                    track.last_confirmed_plate = consensus_plate;
                    track.last_confirmed_confidence = conf.score;
                    track.last_confirmed_frame = frame_number;
                    should_emit_confirmed = false;
                  }

                  if (should_emit_locked &&
                      !should_emit_source_plate_event(video_source,
                                                      "LOCKED",
                                                      consensus_plate,
                                                      conf.score,
                                                      frame_number)) {
                    track.locked = true;
                    track.locked_plate = consensus_plate;
                    track.locked_confidence = conf.score;
                    track.lock_frame = frame_number;
                    track.last_locked_plate = consensus_plate;
                    track.last_locked_confidence = conf.score;
                    track.last_locked_frame = frame_number;
                    should_emit_locked = false;
                  }

                  if (!should_emit_confirmed && !should_emit_locked) {
                    continue;
                  }

                  int veh_left = vehicle_meta ? static_cast<int>(vehicle_meta->rect_params.left) : 0;
                  int veh_top = vehicle_meta ? static_cast<int>(vehicle_meta->rect_params.top) : 0;
                  int veh_width = vehicle_meta ? static_cast<int>(vehicle_meta->rect_params.width) : 0;
                  int veh_height = vehicle_meta ? static_cast<int>(vehicle_meta->rect_params.height) : 0;
                  int plate_left = static_cast<int>(obj_meta->rect_params.left);
                  int plate_top = static_cast<int>(obj_meta->rect_params.top);
                  int plate_width = static_cast<int>(obj_meta->rect_params.width);
                  int plate_height = static_cast<int>(obj_meta->rect_params.height);

                  if (should_emit_confirmed) {
                    log_plate_latency_event("CONFIRMED",
                                            video_source,
                                            consensus_plate,
                                            track,
                                            frame_number,
                                            now,
                                            conf.score,
                                            best_votes,
                                            track.stable_frames);
                    bool confirmed_ok = write_evidence_event(
                      get_frame_mat(),
                      g_evidence_root,
                      video_source,
                      kModelVersion,
                      "CONFIRMED",
                      consensus_plate,
                      vehicle_attributes.make,
                      vehicle_attributes.type,
                      vehicle_attributes.color,
                      conf.score,
                      frame_number,
                      track_id_valid,
                      track_id,
                      veh_left, veh_top, veh_width, veh_height,
                      plate_left, plate_top, plate_width, plate_height,
                      conf.ca_pattern_name);

                    std::cout << "CONFIRMED Plate: " << consensus_plate
                              << " [confidence=" << conf.score << "]";
                    if (!confirmed_ok) std::cout << " [evidence write failed]";
                    std::cout << std::endl;

                    track.reported = true;
                    if (track.first_confirmed_frame < 0) {
                      track.first_confirmed_frame = frame_number;
                      track.first_confirmed_time_us = now;
                    }
                    track.last_confirmed_plate = consensus_plate;
                    track.last_confirmed_confidence = conf.score;
                    track.last_confirmed_frame = frame_number;
                  }

                  if (should_emit_locked) {
                    log_plate_latency_event("LOCKED",
                                            video_source,
                                            consensus_plate,
                                            track,
                                            frame_number,
                                            now,
                                            conf.score,
                                            best_votes,
                                            track.stable_frames);
                    bool locked_ok = write_evidence_event(
                      get_frame_mat(),
                      g_evidence_root,
                      video_source,
                      kModelVersion,
                      "LOCKED",
                      consensus_plate,
                      vehicle_attributes.make,
                      vehicle_attributes.type,
                      vehicle_attributes.color,
                      conf.score,
                      frame_number,
                      track_id_valid,
                      track_id,
                      veh_left, veh_top, veh_width, veh_height,
                      plate_left, plate_top, plate_width, plate_height,
                      conf.ca_pattern_name);

                    std::cout << "LOCKED Plate: " << consensus_plate
                              << " [confidence=" << conf.score << "]";
                    if (!locked_ok) std::cout << " [evidence write failed]";
                    std::cout << std::endl;

                    track.locked = true;
                    track.locked_plate = consensus_plate;
                    track.locked_confidence = conf.score;
                    track.lock_frame = frame_number;
                    track.last_locked_plate = consensus_plate;
                    track.last_locked_confidence = conf.score;
                    track.last_locked_frame = frame_number;
                  }
                }
              }
            }
          }
        }
      }
    }

    if (preview_due) {
      update_runtime_source_preview(get_frame_mat(), video_source, frame_preview_detections);
    }

    display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
    NvOSD_TextParams *txt_params = &display_meta->text_params[0];
    display_meta->num_labels = 1;
    txt_params->display_text = (char *)g_malloc0(MAX_DISPLAY_LEN);
    offset = snprintf(txt_params->display_text, MAX_DISPLAY_LEN, "Person = %d ",
                      person_count);
    offset += snprintf(txt_params->display_text + offset, MAX_DISPLAY_LEN,
                       "Vehicle = %d ", vehicle_count);

    /* Now set the offsets where the string should appear */
    txt_params->x_offset = 10;
    txt_params->y_offset = 12;

    /* Font , font-color and font-size */
    char font_n[6];
    snprintf(font_n, 6, "Serif");
    txt_params->font_params.font_name = font_n;
    txt_params->font_params.font_size = 10;
    txt_params->font_params.font_color.red = 1.0;
    txt_params->font_params.font_color.green = 1.0;
    txt_params->font_params.font_color.blue = 1.0;
    txt_params->font_params.font_color.alpha = 1.0;

    /* Text background color */
    txt_params->set_bg_clr = 1;
    txt_params->text_bg_clr.red = 0.0;
    txt_params->text_bg_clr.green = 0.0;
    txt_params->text_bg_clr.blue = 0.0;
    txt_params->text_bg_clr.alpha = 1.0;

    nvds_add_display_meta_to_frame(frame_meta, display_meta);
  }

  if (verbose_frame_logs_enabled()) {
    g_print("Frame Number = %d Vehicle Count = %d Person Count = %d"
            " License Plate Count = %d\n",
            frame_number, vehicle_count, person_count, lp_count);
  }
  frame_number++;
  total_plate_number += lp_count;
  return GST_PAD_PROBE_OK;
}

static gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer data) {
  GMainLoop *loop = (GMainLoop *)data;
  switch (GST_MESSAGE_TYPE(msg)) {
  case GST_MESSAGE_EOS:
    g_print("End of stream\n");
    g_main_loop_quit(loop);
    break;
  case GST_MESSAGE_ERROR: {
    gchar *debug;
    GError *error;
    gst_message_parse_error(msg, &error, &debug);
    g_printerr("ERROR from element %s: %s\n", GST_OBJECT_NAME(msg->src),
               error->message);
    if (debug)
      g_printerr("Error details: %s\n", debug);
    g_free(debug);
    g_error_free(error);
    g_main_loop_quit(loop);
    break;
  }
  default:
    break;
  }
  return TRUE;
}

static gboolean signal_quit_handler(gpointer data) {
  GMainLoop *loop = (GMainLoop *)data;
  g_print("Termination signal received, stopping main loop\n");
  g_main_loop_quit(loop);
  return G_SOURCE_CONTINUE;
}

static void cb_new_pad(GstElement *element, GstPad *pad, GstElement *data) {
  GstCaps *new_pad_caps = NULL;
  GstStructure *new_pad_struct = NULL;
  const gchar *new_pad_type = NULL;
  GstPadLinkReturn ret;

  GstPad *sink_pad = gst_element_get_static_pad(data, "sink");
  if (gst_pad_is_linked(sink_pad)) {
    g_print("h264parser already linked. Ignoring.\n");
    goto exit;
  }

  new_pad_caps = gst_pad_get_current_caps(pad);
  new_pad_struct = gst_caps_get_structure(new_pad_caps, 0);
  new_pad_type = gst_structure_get_name(new_pad_struct);
  g_print("qtdemux pad %s\n", new_pad_type);

  if (g_str_has_prefix(new_pad_type, "video/x-h264")) {
    ret = gst_pad_link(pad, sink_pad);
    if (GST_PAD_LINK_FAILED(ret))
      g_print("fail to link parser and mp4 demux.\n");
  } else {
    g_print("%s output, not 264 stream\n", new_pad_type);
  }

exit:
  gst_object_unref(sink_pad);
}

static void cb_uri_new_pad(GstElement *element, GstPad *pad, GstElement *data) {
  GstCaps *new_pad_caps = NULL;
  GstStructure *new_pad_struct = NULL;
  const gchar *new_pad_type = NULL;
  GstPadLinkReturn ret;

  GstPad *sink_pad = gst_element_get_static_pad(GST_ELEMENT(data), "sink");
  if (gst_pad_is_linked(sink_pad)) {
    g_print("URI source already linked. Ignoring.\n");
    goto exit;
  }

  new_pad_caps = gst_pad_get_current_caps(pad);
  if (!new_pad_caps) {
    new_pad_caps = gst_pad_query_caps(pad, NULL);
  }
  if (!new_pad_caps) {
    g_print("URI source pad has no caps. Ignoring.\n");
    goto exit;
  }

  new_pad_struct = gst_caps_get_structure(new_pad_caps, 0);
  new_pad_type = gst_structure_get_name(new_pad_struct);
  g_print("uri decodebin pad %s\n", new_pad_type);

  if (g_str_has_prefix(new_pad_type, "video/")) {
    ret = gst_pad_link(pad, sink_pad);
    if (GST_PAD_LINK_FAILED(ret))
      g_print("fail to link uri decodebin to downstream queue.\n");
  } else {
    g_print("%s output, not video stream\n", new_pad_type);
  }

exit:
  if (new_pad_caps) {
    gst_caps_unref(new_pad_caps);
  }
  gst_object_unref(sink_pad);
}

static void set_gobject_property_if_present(GObject *object,
                                            const char *property_name,
                                            int value) {
  if (!object || !property_name) {
    return;
  }
  if (!g_object_class_find_property(G_OBJECT_GET_CLASS(object), property_name)) {
    return;
  }
  g_object_set(object, property_name, value, NULL);
}

static void set_gobject_bool_property_if_present(GObject *object,
                                                 const char *property_name,
                                                 gboolean value) {
  if (!object || !property_name) {
    return;
  }
  if (!g_object_class_find_property(G_OBJECT_GET_CLASS(object), property_name)) {
    return;
  }
  g_object_set(object, property_name, value, NULL);
}

static void cb_uri_source_setup(GstElement * /*uridecodebin*/,
                                GstElement *source,
                                gpointer /*user_data*/) {
  if (!source) {
    return;
  }

  const gchar *factory_name = NULL;
  GstElementFactory *factory = gst_element_get_factory(source);
  if (factory) {
    factory_name = gst_plugin_feature_get_name(GST_PLUGIN_FEATURE(factory));
  }

  if (!factory_name || g_strcmp0(factory_name, "rtspsrc") != 0) {
    return;
  }

  set_gobject_property_if_present(G_OBJECT(source), "latency",
                                  g_runtime_config.rtsp_latency_ms);
  set_gobject_bool_property_if_present(G_OBJECT(source), "drop-on-latency",
                                       g_runtime_config.rtsp_drop_on_latency);
  set_gobject_property_if_present(G_OBJECT(source), "protocols",
                                  g_runtime_config.rtsp_protocols);
  set_gobject_bool_property_if_present(G_OBJECT(source), "do-rtsp-keep-alive", TRUE);

  g_print("configured uridecodebin rtspsrc latency=%d drop_on_latency=%d protocols=%d\n",
          g_runtime_config.rtsp_latency_ms,
          g_runtime_config.rtsp_drop_on_latency ? 1 : 0,
          g_runtime_config.rtsp_protocols);
}

static void cb_rtsp_new_pad(GstElement *element, GstPad *pad, GstElement *data) {
  GstCaps *new_pad_caps = NULL;
  GstStructure *new_pad_struct = NULL;
  const gchar *new_pad_type = NULL;
  GstPadLinkReturn ret;

  GstPad *sink_pad = gst_element_get_static_pad(GST_ELEMENT(data), "sink");
  if (gst_pad_is_linked(sink_pad)) {
    g_print("RTSP depayloader already linked. Ignoring.\n");
    goto exit;
  }

  new_pad_caps = gst_pad_get_current_caps(pad);
  if (!new_pad_caps) {
    new_pad_caps = gst_pad_query_caps(pad, NULL);
  }
  if (!new_pad_caps) {
    g_print("RTSP source pad has no caps. Ignoring.\n");
    goto exit;
  }

  new_pad_struct = gst_caps_get_structure(new_pad_caps, 0);
  new_pad_type = gst_structure_get_name(new_pad_struct);
  g_print("rtspsrc pad %s\n", new_pad_type);

  if (g_str_has_prefix(new_pad_type, "application/x-rtp")) {
    ret = gst_pad_link(pad, sink_pad);
    if (GST_PAD_LINK_FAILED(ret)) {
      g_print("fail to link rtspsrc to rtph264depay.\n");
    }
  } else {
    g_print("%s output, not RTP stream\n", new_pad_type);
  }

exit:
  if (new_pad_caps) {
    gst_caps_unref(new_pad_caps);
  }
  gst_object_unref(sink_pad);
}

/* nvdsanalytics_src_pad_buffer_probe  will extract metadata received on
 * nvdsanalytics src pad and extract nvanalytics metadata etc. */
static GstPadProbeReturn
nvdsanalytics_src_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info,
                                   gpointer u_data) {
  GstBuffer *buf = (GstBuffer *)info->data;
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

  parse_nvdsanalytics_meta_data(batch_meta);

  return GST_PAD_PROBE_OK;
}

static inline const char *infer_plugin(NvDsGieType type) {
  switch (type) {
  case NVDS_GIE_PLUGIN_INFER:
    return "nvinfer";
  case NVDS_GIE_PLUGIN_INFER_SERVER:
    return "nvinferserver";
  default:
    return "unknown";
  }
}

int main(int argc, char *argv[]) {
  GMainLoop *loop = NULL;
  GstElement *pipeline = NULL, *streammux = NULL, *sink = NULL,
             *primary_detector = NULL, *secondary_detector = NULL,
             *nvvidconv = NULL, *nvosd = NULL, *nvvidconv1 = NULL,
             *evidence_capfilt = NULL, *outenc = NULL, *capfilt = NULL,
             *mux = NULL,
             *secondary_classifier = NULL, *vehicle_color_classifier = NULL,
             *vehicle_type_classifier = NULL, *vehicle_make_classifier = NULL,
             *nvtile = NULL, *encparse = NULL;
  GstElement *tracker = NULL, *nvdsanalytics = NULL;
  GstElement *queue1 = NULL, *queue2 = NULL, *queue3 = NULL, *queue4 = NULL,
             *queue5 = NULL, *queue6 = NULL, *queue7 = NULL, *queue8 = NULL,
             *queue9 = NULL, *queue10 = NULL, *queue11 = NULL,
             *queue12 = NULL, *queue13 = NULL;
  GstElement *h264parser[128], *source[128], *decoder[128], *mp4demux[128],
      *rtph264depay[128], *parsequeue[128], *source_convert[128],
      *source_caps[128], *source_nvmm_caps[128];
  GstBus *bus = NULL;
  guint bus_watch_id;
  guint sigint_watch_id = 0;
  guint sigterm_watch_id = 0;
  // int i;
  static guint src_cnt = 0;
  guint tiler_rows, tiler_columns;
  perf_measure perf_measure;

  gchar ele_name[64];
  GstPad *sinkpad, *srcpad;
  gchar pad_name_sink[16] = "sink_0";
  gchar pad_name_src[16] = "src";

  bool isH264 = true;
  bool has_live_source = false;
  int enc_type = ENCODER_TYPE_HW;
  GList *g_list = NULL;
  GList *iterator = NULL;

  NvDsGieType pgie_type = NVDS_GIE_PLUGIN_INFER;
  NvDsGieType sgie0_type = NVDS_GIE_PLUGIN_INFER;
  NvDsGieType sgie1_type = NVDS_GIE_PLUGIN_INFER;
  NvDsGieType sgie2_type = NVDS_GIE_PLUGIN_INFER;
  NvDsGieType sgie3_type = NVDS_GIE_PLUGIN_INFER;
  NvDsGieType sgie4_type = NVDS_GIE_PLUGIN_INFER;

  if (argc != 2) {
    g_printerr("Usage: %s <yml file>\n", argv[0]);
    return -1;
  }

  int current_device = -1;
  cudaGetDevice(&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, current_device);
  set_case_review_index_generator(argv[0]);
  apply_live_dashboard_config(argv[1]);
  g_runtime_config = load_alpr_runtime_config(argv[1]);
  g_print("ALPR runtime profile=%s low_latency_rtsp=%d rtsp_latency_ms=%d "
          "rtsp_drop_on_latency=%d rtsp_protocols=%d latency_metrics=%d\n",
          g_runtime_config.profile.c_str(),
          g_runtime_config.low_latency_rtsp ? 1 : 0,
          g_runtime_config.rtsp_latency_ms,
          g_runtime_config.rtsp_drop_on_latency ? 1 : 0,
          g_runtime_config.rtsp_protocols,
          g_runtime_config.latency_metrics ? 1 : 0);

  // For Chinese language supporting
  setlocale(LC_CTYPE, "");
  /* Standard GStreamer initialization */
  gst_init(&argc, &argv);
  loop = g_main_loop_new(NULL, FALSE);

  perf_measure.pre_time = GST_CLOCK_TIME_NONE;
  perf_measure.total_time = GST_CLOCK_TIME_NONE;
  perf_measure.count = 0;

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new("pipeline");

  /* Create nvstreammux instance to form batches from one or more sources. */
  streammux = gst_element_factory_make("nvstreammux", "stream-muxer");

  if (!pipeline || !streammux) {
    g_printerr("One element could not be created. Exiting.\n");
    return -1;
  }

  gst_bin_add(GST_BIN(pipeline), streammux);

  RETURN_ON_PARSER_ERROR(
      nvds_parse_source_list(&g_list, argv[1], "source-list"));

  std::string startup_video_source = "unknown";
  if (g_list && g_list->data) {
    startup_video_source = static_cast<const char *>(g_list->data);
    if (is_v4l2_camera_source(startup_video_source)) {
      startup_video_source = normalize_v4l2_camera_source(startup_video_source);
    }
  }
  g_evidence_root = std::string("evidence/") + next_case_id();
  if (!ensure_session_json(g_evidence_root, startup_video_source, kModelVersion,
                           REJECT_THRESHOLD, 85)) {
    g_printerr("Failed to initialize evidence/session.json\n");
  }

  RETURN_ON_PARSER_ERROR(
      nvds_parse_gie_type(&pgie_type, argv[1], "primary-gie"));
  RETURN_ON_PARSER_ERROR(
      nvds_parse_gie_type(&sgie0_type, argv[1], "secondary-gie0"));
  RETURN_ON_PARSER_ERROR(
      nvds_parse_gie_type(&sgie1_type, argv[1], "secondary-gie1"));

  bool enable_vehicle_color_classifier = config_has_section(argv[1], "secondary-gie2");
  bool enable_vehicle_type_classifier = config_has_section(argv[1], "secondary-gie3");
  bool enable_vehicle_make_classifier = config_has_section(argv[1], "secondary-gie4");

  if (enable_vehicle_color_classifier) {
    RETURN_ON_PARSER_ERROR(
        nvds_parse_gie_type(&sgie2_type, argv[1], "secondary-gie2"));
  }
  if (enable_vehicle_type_classifier) {
    RETURN_ON_PARSER_ERROR(
        nvds_parse_gie_type(&sgie3_type, argv[1], "secondary-gie3"));
  }
  if (enable_vehicle_make_classifier) {
    RETURN_ON_PARSER_ERROR(
        nvds_parse_gie_type(&sgie4_type, argv[1], "secondary-gie4"));
  }

  /* Multiple source files */
  for (iterator = g_list, src_cnt = 0; iterator;
       iterator = iterator->next, src_cnt++) {
    std::string source_path = static_cast<const char *>(iterator->data);
    bool is_camera_source = is_v4l2_camera_source(source_path);
    bool is_low_latency_rtsp_source =
      !is_camera_source && is_rtsp_uri_source(source_path) &&
      g_runtime_config.low_latency_rtsp;
    bool is_generic_uri_source = !is_camera_source && is_uri_source(source_path);

    h264parser[src_cnt] = NULL;
    rtph264depay[src_cnt] = NULL;
    mp4demux[src_cnt] = NULL;
    decoder[src_cnt] = NULL;
    parsequeue[src_cnt] = NULL;
    source_convert[src_cnt] = NULL;
    source_caps[src_cnt] = NULL;
    source_nvmm_caps[src_cnt] = NULL;

    if (is_camera_source) {
      has_live_source = true;

      g_snprintf(ele_name, 64, "camera_src_%d", src_cnt);
      source[src_cnt] = gst_element_factory_make("v4l2src", ele_name);

      g_snprintf(ele_name, 64, "camera_caps_%d", src_cnt);
      source_caps[src_cnt] = gst_element_factory_make("capsfilter", ele_name);

      g_snprintf(ele_name, 64, "camera_convert_%d", src_cnt);
      source_convert[src_cnt] = gst_element_factory_make("videoconvert", ele_name);

      g_snprintf(ele_name, 64, "camera_queue_%d", src_cnt);
      parsequeue[src_cnt] = gst_element_factory_make("queue", ele_name);

      g_snprintf(ele_name, 64, "camera_nvvidconv_%d", src_cnt);
      decoder[src_cnt] = gst_element_factory_make("nvvideoconvert", ele_name);

      g_snprintf(ele_name, 64, "camera_nvmm_caps_%d", src_cnt);
      source_nvmm_caps[src_cnt] = gst_element_factory_make("capsfilter", ele_name);

      if (!source[src_cnt] || !source_caps[src_cnt] || !source_convert[src_cnt] ||
          !parsequeue[src_cnt] || !decoder[src_cnt] || !source_nvmm_caps[src_cnt]) {
        g_printerr("One camera element could not be created. Exiting.\n");
        return -1;
      }

      gst_bin_add_many(GST_BIN(pipeline), source[src_cnt], source_caps[src_cnt],
                       source_convert[src_cnt], parsequeue[src_cnt], decoder[src_cnt],
                       source_nvmm_caps[src_cnt], NULL);

        gchar *camera_caps_str = g_strdup_printf(
          "video/x-raw,format=YUY2,width=%d,height=%d,framerate=30/1",
          CAMERA_SOURCE_WIDTH, CAMERA_SOURCE_HEIGHT);
        GstCaps *camera_caps = gst_caps_from_string(camera_caps_str);
      GstCaps *camera_nvmm_caps =
          gst_caps_from_string("video/x-raw(memory:NVMM),format=NV12");
      g_object_set(G_OBJECT(source_caps[src_cnt]), "caps", camera_caps, NULL);
      g_object_set(G_OBJECT(source_nvmm_caps[src_cnt]), "caps", camera_nvmm_caps,
                   NULL);
        g_free(camera_caps_str);
      gst_caps_unref(camera_caps);
      gst_caps_unref(camera_nvmm_caps);

      std::string device_path = normalize_v4l2_camera_source(source_path);
      g_object_set(G_OBJECT(source[src_cnt]), "device", device_path.c_str(),
                   "do-timestamp", TRUE, NULL);
      if (!prop.integrated) {
        g_object_set(G_OBJECT(decoder[src_cnt]), "nvbuf-memory-type", 3, NULL);
      }
    } else if (is_low_latency_rtsp_source) {
      has_live_source = true;

      g_snprintf(ele_name, 64, "rtsp_src_%d", src_cnt);
      source[src_cnt] = gst_element_factory_make("rtspsrc", ele_name);

      g_snprintf(ele_name, 64, "rtsp_h264depay_%d", src_cnt);
      rtph264depay[src_cnt] = gst_element_factory_make("rtph264depay", ele_name);

      g_snprintf(ele_name, 64, "rtsp_h264parse_%d", src_cnt);
      h264parser[src_cnt] = gst_element_factory_make("h264parse", ele_name);

      g_snprintf(ele_name, 64, "rtsp_parsequeue_%d", src_cnt);
      parsequeue[src_cnt] = gst_element_factory_make("queue", ele_name);

      g_snprintf(ele_name, 64, "rtsp_decoder_%d", src_cnt);
      decoder[src_cnt] = gst_element_factory_make("nvv4l2decoder", ele_name);

      g_snprintf(ele_name, 64, "rtsp_nvvidconv_%d", src_cnt);
      source_convert[src_cnt] = gst_element_factory_make("nvvideoconvert", ele_name);

      g_snprintf(ele_name, 64, "rtsp_nvmm_caps_%d", src_cnt);
      source_nvmm_caps[src_cnt] = gst_element_factory_make("capsfilter", ele_name);

      if (!source[src_cnt] || !rtph264depay[src_cnt] || !h264parser[src_cnt] ||
          !parsequeue[src_cnt] || !decoder[src_cnt] || !source_convert[src_cnt] ||
          !source_nvmm_caps[src_cnt]) {
        g_printerr("One low-latency RTSP source element could not be created. Exiting.\n");
        return -1;
      }

      gst_bin_add_many(GST_BIN(pipeline), source[src_cnt], rtph264depay[src_cnt],
                       h264parser[src_cnt], parsequeue[src_cnt], decoder[src_cnt],
                       source_convert[src_cnt], source_nvmm_caps[src_cnt], NULL);

      GstCaps *rtsp_nvmm_caps =
          gst_caps_from_string("video/x-raw(memory:NVMM),format=NV12");
      g_object_set(G_OBJECT(source_nvmm_caps[src_cnt]), "caps", rtsp_nvmm_caps,
                   NULL);
      gst_caps_unref(rtsp_nvmm_caps);

      g_object_set(G_OBJECT(source[src_cnt]), "location", source_path.c_str(),
                   "latency", g_runtime_config.rtsp_latency_ms,
                   "drop-on-latency", g_runtime_config.rtsp_drop_on_latency,
                   "protocols", g_runtime_config.rtsp_protocols,
                   "do-rtsp-keep-alive", TRUE, NULL);
      g_signal_connect(source[src_cnt], "pad-added", G_CALLBACK(cb_rtsp_new_pad),
                       rtph264depay[src_cnt]);

      if (!gst_element_link_many(rtph264depay[src_cnt], h264parser[src_cnt],
                                 parsequeue[src_cnt], decoder[src_cnt],
                                 source_convert[src_cnt], source_nvmm_caps[src_cnt],
                                 NULL)) {
        g_printerr("Low-latency RTSP source elements could not be linked. Exiting.\n");
        return -1;
      }

      if (!prop.integrated) {
        g_object_set(G_OBJECT(decoder[src_cnt]), "nvbuf-memory-type", 3, NULL);
        g_object_set(G_OBJECT(source_convert[src_cnt]), "nvbuf-memory-type", 3, NULL);
      }
    } else if (is_generic_uri_source) {
      if (is_live_uri_source(source_path)) {
        has_live_source = true;
      }

      g_snprintf(ele_name, 64, "uri_decode_%d", src_cnt);
      source[src_cnt] = gst_element_factory_make("uridecodebin", ele_name);

      g_snprintf(ele_name, 64, "uri_queue_%d", src_cnt);
      parsequeue[src_cnt] = gst_element_factory_make("queue", ele_name);

      g_snprintf(ele_name, 64, "uri_nvvidconv_%d", src_cnt);
      decoder[src_cnt] = gst_element_factory_make("nvvideoconvert", ele_name);

      g_snprintf(ele_name, 64, "uri_nvmm_caps_%d", src_cnt);
      source_nvmm_caps[src_cnt] = gst_element_factory_make("capsfilter", ele_name);

      if (!source[src_cnt] || !parsequeue[src_cnt] || !decoder[src_cnt] ||
          !source_nvmm_caps[src_cnt]) {
        g_printerr("One URI source element could not be created. Exiting.\n");
        return -1;
      }

      gst_bin_add_many(GST_BIN(pipeline), source[src_cnt], parsequeue[src_cnt],
                       decoder[src_cnt], source_nvmm_caps[src_cnt], NULL);

      /* Keep patrol display/read latency bounded on live IP cameras. This
       * queue receives decoded raw video from uridecodebin, so dropping here
       * avoids H.264 corruption while preventing a stale-frame backlog. */
      if (is_live_uri_source(source_path)) {
        g_object_set(G_OBJECT(parsequeue[src_cnt]),
                     "max-size-buffers", 2,
                     "max-size-bytes", 0,
                     "max-size-time", 0,
                     "leaky", 2,
                     NULL);
      }

      GstCaps *uri_nvmm_caps =
          gst_caps_from_string("video/x-raw(memory:NVMM),format=NV12");
      g_object_set(G_OBJECT(source_nvmm_caps[src_cnt]), "caps", uri_nvmm_caps,
                   NULL);
      gst_caps_unref(uri_nvmm_caps);

      g_object_set(G_OBJECT(source[src_cnt]), "uri", source_path.c_str(), NULL);
      g_signal_connect(source[src_cnt], "source-setup",
                       G_CALLBACK(cb_uri_source_setup), NULL);
      g_signal_connect(source[src_cnt], "pad-added", G_CALLBACK(cb_uri_new_pad),
                       parsequeue[src_cnt]);

      if (!gst_element_link_many(parsequeue[src_cnt], decoder[src_cnt],
                                 source_nvmm_caps[src_cnt], NULL)) {
        g_printerr("URI source elements could not be linked. Exiting.\n");
        return -1;
      }

      if (!prop.integrated) {
        g_object_set(G_OBJECT(decoder[src_cnt]), "nvbuf-memory-type", 3, NULL);
      }
    } else {
      /* Only h264 element stream with mp4 container is supported. */
      g_snprintf(ele_name, 64, "file_src_%d", src_cnt);

      /* Source element for reading from the file */
      source[src_cnt] = gst_element_factory_make("filesrc", ele_name);

      g_snprintf(ele_name, 64, "mp4demux_%d", src_cnt);
      mp4demux[src_cnt] = gst_element_factory_make("qtdemux", ele_name);

      g_snprintf(ele_name, 64, "h264parser_%d", src_cnt);
      h264parser[src_cnt] = gst_element_factory_make("h264parse", ele_name);

      g_snprintf(ele_name, 64, "parsequeue_%d", src_cnt);
      parsequeue[src_cnt] = gst_element_factory_make("queue", ele_name);

      /* Use nvdec_h264 for hardware accelerated decode on GPU */
      g_snprintf(ele_name, 64, "decoder_%d", src_cnt);
      decoder[src_cnt] = gst_element_factory_make("nvv4l2decoder", ele_name);

      if (!source[src_cnt] || !h264parser[src_cnt] || !decoder[src_cnt] ||
          !mp4demux[src_cnt]) {
        g_printerr("One element could not be created. Exiting.\n");
        return -1;
      }

      gst_bin_add_many(GST_BIN(pipeline), source[src_cnt], mp4demux[src_cnt],
                       h264parser[src_cnt], parsequeue[src_cnt], decoder[src_cnt],
                       NULL);

      /* we set the input filename to the source element */
      g_object_set(G_OBJECT(source[src_cnt]), "location", (gchar *)iterator->data,
                   NULL);
    }

    g_snprintf(pad_name_sink, 64, "sink_%d", src_cnt);
    sinkpad = gst_element_get_request_pad(streammux, pad_name_sink);;
    g_print("Request %s pad from streammux\n", pad_name_sink);
    if (!sinkpad) {
      g_printerr("Streammux request sink pad failed. Exiting.\n");
      return -1;
    }

    srcpad = gst_element_get_static_pad(
      (is_camera_source || is_generic_uri_source) ? source_nvmm_caps[src_cnt]
                            : decoder[src_cnt],
      pad_name_src);
    if (!srcpad) {
      g_printerr("Source request src pad failed. Exiting.\n");
      return -1;
    }

    if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK) {
      g_printerr("Failed to link decoder to stream muxer. Exiting.\n");
      return -1;
    }

    if (is_camera_source) {
      if (!gst_element_link_many(source[src_cnt], source_caps[src_cnt],
                                 source_convert[src_cnt], parsequeue[src_cnt],
                                 decoder[src_cnt], source_nvmm_caps[src_cnt],
                                 NULL)) {
        g_printerr("Camera elements could not be linked. Exiting.\n");
        return -1;
      }
    } else if (!is_generic_uri_source) {
      if (!gst_element_link_pads(source[src_cnt], "src", mp4demux[src_cnt],
                                 "sink")) {
        g_printerr("Elements could not be linked: 0. Exiting.\n");
        return -1;
      }

      g_signal_connect(mp4demux[src_cnt], "pad-added", G_CALLBACK(cb_new_pad),
                       h264parser[src_cnt]);

      if (!gst_element_link_many(h264parser[src_cnt], parsequeue[src_cnt],
                                 decoder[src_cnt], NULL)) {
        g_printerr("Elements could not be linked: 1. Exiting.\n");
      }
    }

    gst_object_unref(sinkpad);
    gst_object_unref(srcpad);
  }
  g_list_free(g_list);

  /* Create three nvinfer instances for two detectors and one classifier*/
  primary_detector = gst_element_factory_make(infer_plugin(pgie_type),
                                              "primary-infer-engine1");

  secondary_detector = gst_element_factory_make(infer_plugin(sgie0_type),
                                                "secondary-infer-engine1");

  secondary_classifier = gst_element_factory_make(infer_plugin(sgie1_type),
                                                  "secondary-infer-engine2");

  if (enable_vehicle_color_classifier) {
    vehicle_color_classifier = gst_element_factory_make(infer_plugin(sgie2_type),
                                                        "secondary-infer-engine3");
  }
  if (enable_vehicle_type_classifier) {
    vehicle_type_classifier = gst_element_factory_make(infer_plugin(sgie3_type),
                                                       "secondary-infer-engine4");
  }
  if (enable_vehicle_make_classifier) {
    vehicle_make_classifier = gst_element_factory_make(infer_plugin(sgie4_type),
                                                       "secondary-infer-engine5");
  }

  /* Use convertor to convert from NV12 to RGBA as required by nvosd */
  nvvidconv = gst_element_factory_make("nvvideoconvert", "nvvid-converter");
  evidence_capfilt = gst_element_factory_make("capsfilter", "evidence-rgba-caps");

  /* Create OSD to draw on the converted RGBA buffer */
  nvosd = gst_element_factory_make("nvdsosd", "nv-onscreendisplay");

  nvvidconv1 = gst_element_factory_make("nvvideoconvert", "nvvid-converter1");

  capfilt = gst_element_factory_make("capsfilter", "nvvideo-caps");

  nvtile = gst_element_factory_make("nvmultistreamtiler", "nvtiler");

  tracker = gst_element_factory_make("nvtracker", "nvtracker");

  /* Use nvdsanalytics to perform analytics on object */
  nvdsanalytics = NULL;

  queue1 = gst_element_factory_make("queue", "queue1");
  queue2 = gst_element_factory_make("queue", "queue2");
  queue3 = gst_element_factory_make("queue", "queue3");
  queue4 = gst_element_factory_make("queue", "queue4");
  queue5 = gst_element_factory_make("queue", "queue5");
  queue6 = gst_element_factory_make("queue", "queue6");
  queue7 = gst_element_factory_make("queue", "queue7");
  queue8 = gst_element_factory_make("queue", "queue8");
  queue9 = gst_element_factory_make("queue", "queue9");
  queue10 = gst_element_factory_make("queue", "queue10");
  queue11 = gst_element_factory_make("queue", "queue11");
  queue12 = gst_element_factory_make("queue", "queue12");
  queue13 = gst_element_factory_make("queue", "queue13");

  /* set properties for nvdsanalytics */
  // ds_parse_nvdsanalytics(nvdsanalytics, argv[1], "analytics");
  if (nvdsanalytics) {
    g_object_set(G_OBJECT(nvdsanalytics), "enable", FALSE, NULL);
  }

  guint output_type = 2;
  output_type = ds_parse_group_type(argv[1], "output");
  if (output_type == 1) {
    sink = gst_element_factory_make("filesink", "nvvideo-renderer");
  } else if (output_type == 2) {
    sink = gst_element_factory_make("fakesink", "fake-renderer");
  } else if (output_type == 3) {
    if (prop.integrated)
      sink = gst_element_factory_make("nv3dsink", "nv3d-sink");
    else
#ifdef __aarch64__
      sink = gst_element_factory_make("nv3dsink", "nv3d-sink");
#else
      sink = gst_element_factory_make("nveglglessink", "nvvideo-renderer");
#endif
  }

  if (!primary_detector || !secondary_detector || !secondary_classifier || !nvvidconv ||
      !evidence_capfilt || !nvosd || !queue11 || !queue12 || !queue13 ||
      !sink) {
    g_printerr("One element could not be created. Exiting.\n");
    return -1;
  }

  if ((enable_vehicle_color_classifier && !vehicle_color_classifier) ||
      (enable_vehicle_type_classifier && !vehicle_type_classifier) ||
      (enable_vehicle_make_classifier && !vehicle_make_classifier)) {
    g_printerr("One vehicle attribute classifier element could not be created. Exiting.\n");
    return -1;
  }

  g_object_set(G_OBJECT(streammux), "width", MUXER_OUTPUT_WIDTH, "height",
               MUXER_OUTPUT_HEIGHT, "batch-size", src_cnt,
               "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC,
               "live-source", has_live_source, NULL);

  tiler_rows = (guint)sqrt(src_cnt);
  tiler_columns = (guint)ceil(1.0 * src_cnt / tiler_rows);
  g_object_set(G_OBJECT(nvtile), "rows", tiler_rows, "columns", tiler_columns,
               "width", MUXER_OUTPUT_WIDTH, "height", MUXER_OUTPUT_HEIGHT, NULL);

  if (!prop.integrated) {
    g_object_set(G_OBJECT(nvvidconv), "nvbuf-memory-type", 3, NULL);
  }

  GstCaps *evidence_caps =
      gst_caps_from_string("video/x-raw(memory:NVMM), format=RGBA");
  g_object_set(G_OBJECT(evidence_capfilt), "caps", evidence_caps, NULL);
  gst_caps_unref(evidence_caps);

  /* Set the config files for the two detectors and one classifier. The PGIE
   * detects the cars. The first SGIE detects car plates from the cars and the
   * second SGIE classifies the caracters in the car plate to identify the car
   * plate string. */
  nvds_parse_gie(primary_detector, argv[1], "primary-gie");
  nvds_parse_gie(secondary_detector, argv[1], "secondary-gie0");
  nvds_parse_gie(secondary_classifier, argv[1], "secondary-gie1");
  if (enable_vehicle_color_classifier) {
    nvds_parse_gie(vehicle_color_classifier, argv[1], "secondary-gie2");
  }
  if (enable_vehicle_type_classifier) {
    nvds_parse_gie(vehicle_type_classifier, argv[1], "secondary-gie3");
  }
  if (enable_vehicle_make_classifier) {
    nvds_parse_gie(vehicle_make_classifier, argv[1], "secondary-gie4");
  }
  nvds_parse_tracker(tracker, argv[1], "tracker");

  /* we add a bus message handler */
  bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
  bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
  gst_object_unref(bus);
  sigint_watch_id = g_unix_signal_add(SIGINT, signal_quit_handler, loop);
  sigterm_watch_id = g_unix_signal_add(SIGTERM, signal_quit_handler, loop);

  /* Set up the pipeline */
  /* we add all elements into the pipeline */
  gst_bin_add_many(GST_BIN(pipeline), primary_detector, secondary_detector,
                   tracker, queue1, queue2, queue3, queue4,
                   queue5, queue6, queue7, queue8, secondary_classifier,
                   nvvidconv, evidence_capfilt, nvosd, nvtile, NULL);

  std::vector<GstElement*> attribute_classifiers;
  if (enable_vehicle_color_classifier) {
    gst_bin_add(GST_BIN(pipeline), vehicle_color_classifier);
    attribute_classifiers.push_back(vehicle_color_classifier);
  }
  if (enable_vehicle_type_classifier) {
    gst_bin_add(GST_BIN(pipeline), vehicle_type_classifier);
    attribute_classifiers.push_back(vehicle_type_classifier);
  }
  if (enable_vehicle_make_classifier) {
    gst_bin_add(GST_BIN(pipeline), vehicle_make_classifier);
    attribute_classifiers.push_back(vehicle_make_classifier);
  }

  GstElement* attribute_queues[] = {queue11, queue12, queue13};
  for (size_t i = 0; i < attribute_classifiers.size(); ++i) {
    gst_bin_add(GST_BIN(pipeline), attribute_queues[i]);
  }

  std::vector<GstElement*> inference_chain = {
      streammux, queue1, primary_detector, queue2, tracker, queue3, queue4,
      secondary_detector, queue5, secondary_classifier, queue6};

  for (size_t i = 0; i < attribute_classifiers.size(); ++i) {
    inference_chain.push_back(attribute_classifiers[i]);
    inference_chain.push_back(attribute_queues[i]);
  }

  inference_chain.push_back(nvtile);
  inference_chain.push_back(queue7);
  inference_chain.push_back(nvvidconv);
  inference_chain.push_back(evidence_capfilt);
  inference_chain.push_back(queue8);
  inference_chain.push_back(nvosd);

  if (!link_element_chain(inference_chain)) {
    g_printerr("Inferring and tracking elements link failure.\n");
    return -1;
  }

  if (output_type == 1) {
    isH264 = !(ds_parse_enc_codec(argv[1], "output"));
    enc_type = ds_parse_enc_type(argv[1], "output");
    create_video_encoder(isH264, enc_type, &capfilt, &outenc, &encparse, NULL);
    if (!capfilt || !outenc || !encparse) {
      g_printerr("enc element could not be created. Exiting.\n");
      return -1;
    }
    gchar *filepath = NULL;
    mux = gst_element_factory_make("qtmux", "mp4-mux");

    GString *output_file = ds_parse_file_name(argv[1], "output");
    filepath = g_strconcat(output_file->str, ".mp4", NULL);
    ds_parse_enc_config(outenc, argv[1], "output");

    g_object_set(G_OBJECT(sink), "async", FALSE, NULL);
    g_object_set(G_OBJECT(sink), "sync", TRUE, NULL);
    g_object_set(G_OBJECT(sink), "location", filepath, NULL);
    gst_bin_add_many(GST_BIN(pipeline), queue9, nvvidconv1, capfilt, queue10,
                     outenc, encparse, mux, sink, NULL);

    if (!gst_element_link_many(nvosd, queue9, nvvidconv1, capfilt, queue10,
                               outenc, encparse, mux, sink, NULL)) {
      g_printerr("OSD and sink elements link failure.\n");
      return -1;
    }
  } else if (output_type == 2) {
    gst_bin_add(GST_BIN(pipeline), sink);
    g_object_set(G_OBJECT(sink), "sync", 0, "async", false, NULL);
    if (!gst_element_link(nvosd, sink)) {
      g_printerr("OSD and sink elements link failure.\n");
      return -1;
    }
  } else if (output_type == 3) {
    gst_bin_add_many(GST_BIN(pipeline), queue9, sink, NULL);
    if (!gst_element_link_many(nvosd, queue9, sink, NULL)) {
      g_printerr("OSD and sink elements link failure.\n");
      return -1;
    }
  }

  /* Lets add probe to get informed of the meta data generated, we add probe to
   * the sink pad of the osd element, since by that time, the buffer would have
   * had got all the metadata. */
  GstPad *evidence_sink_pad = gst_element_get_static_pad(evidence_capfilt, "sink");
  if (!evidence_sink_pad) {
    g_print("Unable to get evidence sink pad\n");
  } else {
    gst_pad_add_probe(evidence_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
                      osd_sink_pad_buffer_probe, &perf_measure, NULL);
    gst_object_unref(evidence_sink_pad);
  }

  /* Pre-LPD large-vehicle crop probe: focus tall/wide vehicle bounding boxes on
   * the bottom plate zone before LPDNet secondary GIE crops them for inference.
   * queue4 src pad sits between the tracker and secondary_detector (LPDNet). */
  GstPad *pre_lpd_pad = gst_element_get_static_pad(queue4, "src");
  if (!pre_lpd_pad) {
    g_print("Unable to get queue4 src pad for pre-LPD large-vehicle probe\n");
  } else {
    gst_pad_add_probe(pre_lpd_pad, GST_PAD_PROBE_TYPE_BUFFER,
                      pre_lpd_large_vehicle_crop_probe, NULL, NULL);
    gst_object_unref(pre_lpd_pad);
    g_print("Pre-LPD large-vehicle crop probe attached (queue4 src)\n");
  }

  /* Post-LPR restore probe: restores original full-vehicle bounding boxes after
   * LPDNet+LPRNet have run, so attribute classifiers (make/type/color) receive
   * the correct full-vehicle crop.
   * queue6 src pad sits between secondary_classifier (LPRNet) and the first
   * attribute classifier. */
  GstPad *post_lpr_pad = gst_element_get_static_pad(queue6, "src");
  if (!post_lpr_pad) {
    g_print("Unable to get queue6 src pad for post-LPR restore probe\n");
  } else {
    gst_pad_add_probe(post_lpr_pad, GST_PAD_PROBE_TYPE_BUFFER,
                      post_lpr_restore_bbox_probe, NULL, NULL);
    gst_object_unref(post_lpr_pad);
    g_print("Post-LPR bbox restore probe attached (queue6 src)\n");
  }

  //osd_sink_pad = gst_element_get_static_pad(nvdsanalytics, "src");
  //if (!osd_sink_pad)
  //  g_print("Unable to get src pad\n");
  //else
  //  gst_pad_add_probe(osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
  //                    nvdsanalytics_src_pad_buffer_probe, NULL, NULL);
  //gst_object_unref(osd_sink_pad);

  /* Set the pipeline to "playing" state */
  g_print("Now playing: %s\n", argv[1]);
  gst_element_set_state(pipeline, GST_STATE_PLAYING);

  /* Wait till pipeline encounters an error or EOS */
  g_print("Running...\n");
  g_main_loop_run(loop);

  /* Out of the main loop, clean up nicely */
  g_print("Returned, stopping playback\n");
  gst_element_set_state(pipeline, GST_STATE_NULL);

  g_print("Average fps %f\n", ((perf_measure.count - 1) * src_cnt * 1000000.0) /
                                  perf_measure.total_time);
  g_print("Totally %d plates are inferred\n", total_plate_number);
  if (!g_evidence_root.empty()) {
    bool index_ok = generate_case_review_index(g_evidence_root, argv[0]);
    if (index_ok) {
      g_print("Case review index generated in %s\n", g_evidence_root.c_str());
    } else {
      g_printerr("Failed to generate case review index for %s\n", g_evidence_root.c_str());
    }
  }
  g_print("Deleting pipeline\n");
  gst_object_unref(GST_OBJECT(pipeline));
  g_source_remove(bus_watch_id);
  if (sigint_watch_id != 0) {
    g_source_remove(sigint_watch_id);
  }
  if (sigterm_watch_id != 0) {
    g_source_remove(sigterm_watch_id);
  }
  g_main_loop_unref(loop);
  return 0;
}
