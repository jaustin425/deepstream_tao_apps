#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cv {
class Mat;
}

struct EvidenceEvent {
    std::string event_id;
    std::string event_type;   // CONFIRMED or LOCKED
    std::string plate;
    std::string vehicle_make;
    std::string vehicle_type;
    std::string vehicle_color;
    int confidence = 0;
    int frame_number = 0;
    bool track_id_valid = false;
    uint64_t track_id = 0;
    std::string video_source;
    std::string timestamp_utc;
    bool gps_fix_valid = false;
    double gps_latitude = 0.0;
    double gps_longitude = 0.0;
    double gps_altitude_m = 0.0;
    double gps_speed_knots = 0.0;
    std::string gps_timestamp_utc;

    int veh_left = 0;
    int veh_top = 0;
    int veh_width = 0;
    int veh_height = 0;

    int plate_left = 0;
    int plate_top = 0;
    int plate_width = 0;
    int plate_height = 0;

    std::string full_frame_path;
    std::string plate_crop_path;
    std::string annotated_frame_path;
    std::string full_frame_sha256;
    std::string plate_crop_sha256;
    std::string annotated_frame_sha256;

    std::string model_version;
    std::string notes;
};

struct RuntimePreviewDetection {
    int left = 0;
    int top = 0;
    int width = 0;
    int height = 0;
    int confidence = 0;
    std::string plate;
    std::string focus_state;
};

bool ensure_dir(const std::string& path);
std::string utc_now_iso8601();
std::string next_event_id();
std::string next_case_id();
std::string current_case_id();
void set_case_review_index_generator(const std::string& executable_path);
std::string sha256_file(const std::string& path);
bool generate_case_review_index(const std::string& case_dir,
                                const std::string& executable_path);
bool ensure_session_json(const std::string& evidence_root,
                         const std::string& video_source,
                         const std::string& model_version,
                         int reject_threshold = -1,
                         int lock_threshold = -1);
bool append_event_jsonl(const std::string& jsonl_path, const EvidenceEvent& ev);
void update_runtime_source_status(const std::string& case_id,
                                  const std::string& video_source,
                                  int frame_number);
bool should_update_runtime_source_preview(const std::string& video_source);
void update_runtime_source_preview(const cv::Mat& frame,
                                   const std::string& video_source,
                                   const std::vector<RuntimePreviewDetection>& detections);
bool save_event_images(const cv::Mat& frame,
                       int plate_left, int plate_top, int plate_width, int plate_height,
                       const std::string& case_id,
                       const std::string& event_id,
                       const std::string& full_frame_path,
                       const std::string& plate_crop_path);
bool write_evidence_event(const cv::Mat& frame,
                          const std::string& evidence_root,
                          const std::string& video_source,
                          const std::string& model_version,
                          const std::string& event_type,
                          const std::string& plate,
                          const std::string& vehicle_make,
                          const std::string& vehicle_type,
                          const std::string& vehicle_color,
                          int confidence,
                          int frame_number,
                          bool track_id_valid,
                          uint64_t track_id,
                          int veh_left, int veh_top, int veh_width, int veh_height,
                          int plate_left, int plate_top, int plate_width, int plate_height);
bool write_debug_event(const cv::Mat& frame,
                              const std::string& evidence_root,
                              const std::string& video_source,
                              const std::string& model_version,
                              const std::string& plate,
                              const std::string& vehicle_make,
                              const std::string& vehicle_type,
                              const std::string& vehicle_color,
                              int confidence,
                              int frame_number,
                              bool track_id_valid,
                              uint64_t track_id,
                              int veh_left, int veh_top, int veh_width, int veh_height,
                              int plate_left, int plate_top, int plate_width, int plate_height);