#include "evidence_logger.hpp"

#include <algorithm>
#include <arpa/inet.h>
#include <atomic>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <netdb.h>
#include <cstdio>
#include <fcntl.h>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <termios.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

static std::mutex g_event_mutex;
static uint64_t g_event_counter = 0;
static std::mutex g_session_mutex;
static bool g_session_written = false;
static std::string g_session_case_id;
static std::string g_session_created_utc;
static std::string g_case_index_script_path;
static std::mutex g_live_log_mutex;
static bool g_live_publish_config_warned = false;
static std::mutex g_runtime_status_mutex;
static std::once_flag g_gps_thread_once;
static std::mutex g_gps_mutex;
static std::atomic<bool> g_gps_thread_started(false);
static const int RUNTIME_PREVIEW_WIDTH = 960;
static const int RUNTIME_PREVIEW_HEIGHT = 600;
static const int RUNTIME_PREVIEW_JPEG_QUALITY = 85;
static const int RUNTIME_PREVIEW_INTERVAL_MS = 500;

struct GpsFix {
    bool valid = false;
    double latitude = 0.0;
    double longitude = 0.0;
    double altitude_m = 0.0;
    double speed_knots = 0.0;
    std::string timestamp_utc;
    std::chrono::steady_clock::time_point received_at =
        std::chrono::steady_clock::time_point::min();
};

static GpsFix g_latest_gps_fix;

static std::string json_number_or_null(bool valid, double value, int precision);
static std::string json_string_or_null(const std::string& value);

struct RuntimeSourceState {
    std::string source;
    std::string last_seen_utc;
    int last_frame_number = 0;
    std::string preview_path;
    std::string preview_updated_utc;
    uint64_t preview_sequence = 0;
    int preview_overlay_width = 0;
    int preview_overlay_height = 0;
    std::vector<RuntimePreviewDetection> preview_detections;
    std::chrono::steady_clock::time_point last_preview_write =
        std::chrono::steady_clock::time_point::min();
};

static std::map<std::string, RuntimeSourceState> g_runtime_sources;
static std::chrono::steady_clock::time_point g_runtime_status_last_flush =
    std::chrono::steady_clock::time_point::min();

struct LiveEndpoint {
    std::string host;
    int port = 80;
    std::string path = "/api/live-event";
    bool valid = false;
};

struct LiveSourceInfo {
    std::string code;
    std::string label;
    bool valid = false;
};

static bool path_exists(const std::string& path) {
    struct stat info;
    return stat(path.c_str(), &info) == 0;
}

bool ensure_dir(const std::string& path) {
    std::string command = "mkdir -p \"" + path + "\"";
    return std::system(command.c_str()) == 0;
}

std::string utc_now_iso8601() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    gmtime_r(&t, &tm);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

std::string next_event_id() {
    std::string case_id = next_case_id();

    std::lock_guard<std::mutex> lock(g_event_mutex);
    ++g_event_counter;
    std::ostringstream oss;
    oss << case_id << "_evt_" << std::setw(6) << std::setfill('0') << g_event_counter;
    return oss.str();
}

std::string next_case_id() {
    std::lock_guard<std::mutex> lock(g_session_mutex);
    if (!g_session_case_id.empty()) {
        return g_session_case_id;
    }

    std::string created = utc_now_iso8601();
    std::string date = created.substr(0, 10);
    std::replace(date.begin(), date.end(), '-', '_');

    ensure_dir("evidence");

    int sequence = 1;
    while (true) {
        std::ostringstream oss;
        oss << "case_" << date << "_" << std::setw(3) << std::setfill('0') << sequence;
        std::string candidate = oss.str();
        if (!path_exists("evidence/" + candidate)) {
            g_session_case_id = candidate;
            break;
        }
        ++sequence;
    }

    g_session_created_utc = created;
    return g_session_case_id;
}

std::string current_case_id() {
    std::lock_guard<std::mutex> lock(g_session_mutex);
    return g_session_case_id;
}

void set_case_review_index_generator(const std::string& executable_path) {
    std::string::size_type pos = executable_path.find_last_of('/');
    std::string script_dir = (pos == std::string::npos) ? "." : executable_path.substr(0, pos);
    if (script_dir.empty()) {
        script_dir = ".";
    }
    g_case_index_script_path = script_dir + "/build_case_index.py";
}

bool generate_case_review_index(const std::string& case_dir,
                                const std::string& executable_path) {
    if (case_dir.empty()) {
        return false;
    }

    auto shell_quote = [](const std::string& value) {
        std::string quoted = "'";
        for (char ch : value) {
            if (ch == '\'') {
                quoted += "'\\''";
            } else {
                quoted.push_back(ch);
            }
        }
        quoted += "'";
        return quoted;
    };

    std::string script_path = g_case_index_script_path;
    if (script_path.empty()) {
        std::string::size_type pos = executable_path.find_last_of('/');
        std::string script_dir = (pos == std::string::npos) ? "." : executable_path.substr(0, pos);
        if (script_dir.empty()) {
            script_dir = ".";
        }
        script_path = script_dir + "/build_case_index.py";
    }

    std::string command = "python3 " + shell_quote(script_path) + " " + shell_quote(case_dir);
    return std::system(command.c_str()) == 0;
}

std::string sha256_file(const std::string& path) {
    std::string command = "sha256sum \"" + path + "\"";
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) return "";

    char buffer[512];
    std::string output;
    if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        output = buffer;
    }
    pclose(pipe);

    std::istringstream iss(output);
    std::string hash;
    iss >> hash;
    return hash;
}

static std::string json_escape(const std::string& s) {
    std::ostringstream oss;
    for (char c : s) {
        switch (c) {
            case '\"': oss << "\\\""; break;
            case '\\': oss << "\\\\"; break;
            case '\b': oss << "\\b"; break;
            case '\f': oss << "\\f"; break;
            case '\n': oss << "\\n"; break;
            case '\r': oss << "\\r"; break;
            case '\t': oss << "\\t"; break;
            default:
                if ((unsigned char)c < 0x20) {
                    oss << "\\u"
                        << std::hex << std::setw(4) << std::setfill('0')
                        << (int)(unsigned char)c;
                } else {
                    oss << c;
                }
        }
    }
    return oss.str();
}

static std::string basename_from_path(const std::string& path) {
    std::string::size_type pos = path.find_last_of('/');
    if (pos == std::string::npos) {
        return path;
    }
    return path.substr(pos + 1);
}

static int clamp_preview_value(int value, int lower, int upper) {
    return std::max(lower, std::min(value, upper));
}

static cv::Size runtime_preview_size_for_frame(const cv::Mat& frame) {
    if (frame.empty() || frame.cols <= 0 || frame.rows <= 0) {
        return cv::Size(RUNTIME_PREVIEW_WIDTH, RUNTIME_PREVIEW_HEIGHT);
    }

    double scale = std::min(
        static_cast<double>(RUNTIME_PREVIEW_WIDTH) / frame.cols,
        static_cast<double>(RUNTIME_PREVIEW_HEIGHT) / frame.rows);
    if (!std::isfinite(scale) || scale <= 0.0) {
        return cv::Size(RUNTIME_PREVIEW_WIDTH, RUNTIME_PREVIEW_HEIGHT);
    }

    int preview_width = std::max(1, static_cast<int>(std::lround(frame.cols * scale)));
    int preview_height = std::max(1, static_cast<int>(std::lround(frame.rows * scale)));
    return cv::Size(preview_width, preview_height);
}

static RuntimePreviewDetection scale_preview_detection(const RuntimePreviewDetection& detection,
                                                      int frame_width,
                                                      int frame_height,
                                                      int preview_width,
                                                      int preview_height) {
    RuntimePreviewDetection scaled = detection;
    if (frame_width <= 0 || frame_height <= 0 ||
        preview_width <= 0 || preview_height <= 0) {
        scaled.left = 0;
        scaled.top = 0;
        scaled.width = 0;
        scaled.height = 0;
        return scaled;
    }

    const double scale_x = static_cast<double>(preview_width) / frame_width;
    const double scale_y = static_cast<double>(preview_height) / frame_height;

    scaled.left = clamp_preview_value(
        static_cast<int>(std::lround(detection.left * scale_x)),
        0,
        preview_width);
    scaled.top = clamp_preview_value(
        static_cast<int>(std::lround(detection.top * scale_y)),
        0,
        preview_height);
    scaled.width = clamp_preview_value(
        static_cast<int>(std::lround(detection.width * scale_x)),
        0,
        preview_width - scaled.left);
    scaled.height = clamp_preview_value(
        static_cast<int>(std::lround(detection.height * scale_y)),
        0,
        preview_height - scaled.top);
    return scaled;
}

static std::string dirname_from_path(const std::string& path) {
    std::string::size_type pos = path.find_last_of('/');
    if (pos == std::string::npos) {
        return ".";
    }
    if (pos == 0) {
        return "/";
    }
    return path.substr(0, pos);
}

static std::string uppercase_copy(const std::string& value) {
    std::string result = value;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::toupper(ch)); });
    return result;
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

static std::string getenv_string(const char* name) {
    const char* value = std::getenv(name);
    return value ? std::string(value) : std::string();
}

static int getenv_int(const char* name, int default_value) {
    std::string raw = trim_copy(getenv_string(name));
    if (raw.empty()) {
        return default_value;
    }
    char* end = nullptr;
    long parsed = std::strtol(raw.c_str(), &end, 10);
    if (!end || *end != '\0') {
        return default_value;
    }
    return parsed > 0 ? static_cast<int>(parsed) : default_value;
}

static speed_t baud_to_termios(int baud_rate) {
    switch (baud_rate) {
        case 4800: return B4800;
        case 9600: return B9600;
        case 19200: return B19200;
        case 38400: return B38400;
        case 57600: return B57600;
        case 115200: return B115200;
        default: return B9600;
    }
}

static std::vector<std::string> split_csv_fields(const std::string& value) {
    std::vector<std::string> fields;
    std::string current;
    for (char ch : value) {
        if (ch == ',') {
            fields.push_back(current);
            current.clear();
            continue;
        }
        current.push_back(ch);
    }
    fields.push_back(current);
    return fields;
}

static bool parse_double_value(const std::string& value, double& output) {
    if (value.empty()) {
        return false;
    }
    char* end = nullptr;
    double parsed = std::strtod(value.c_str(), &end);
    if (!end || *end != '\0') {
        return false;
    }
    output = parsed;
    return true;
}

static bool parse_nmea_coordinate(const std::string& raw_value,
                                  const std::string& hemisphere,
                                  bool latitude,
                                  double& output) {
    double numeric = 0.0;
    if (!parse_double_value(raw_value, numeric)) {
        return false;
    }

    double degrees = 0.0;
    double minutes = 0.0;
    if (latitude) {
        degrees = std::floor(numeric / 100.0);
        minutes = numeric - (degrees * 100.0);
    } else {
        degrees = std::floor(numeric / 100.0);
        minutes = numeric - (degrees * 100.0);
    }

    output = degrees + (minutes / 60.0);
    std::string hemi = uppercase_copy(trim_copy(hemisphere));
    if (hemi == "S" || hemi == "W") {
        output = -output;
    }
    return true;
}

static std::string nmea_date_time_to_utc(const std::string& hhmmss,
                                         const std::string& ddmmyy) {
    if (hhmmss.size() < 6 || ddmmyy.size() != 6) {
        return std::string();
    }

    std::tm tm{};
    tm.tm_mday = std::atoi(ddmmyy.substr(0, 2).c_str());
    tm.tm_mon = std::atoi(ddmmyy.substr(2, 2).c_str()) - 1;
    int year = std::atoi(ddmmyy.substr(4, 2).c_str());
    tm.tm_year = (year >= 80 ? 1900 + year : 2000 + year) - 1900;
    tm.tm_hour = std::atoi(hhmmss.substr(0, 2).c_str());
    tm.tm_min = std::atoi(hhmmss.substr(2, 2).c_str());
    tm.tm_sec = std::atoi(hhmmss.substr(4, 2).c_str());

    std::ostringstream out;
    out << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return out.str();
}

static std::string today_with_nmea_time(const std::string& hhmmss) {
    if (hhmmss.size() < 6) {
        return std::string();
    }

    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    gmtime_r(&t, &tm);
    tm.tm_hour = std::atoi(hhmmss.substr(0, 2).c_str());
    tm.tm_min = std::atoi(hhmmss.substr(2, 2).c_str());
    tm.tm_sec = std::atoi(hhmmss.substr(4, 2).c_str());

    std::ostringstream out;
    out << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return out.str();
}

static void store_latest_gps_fix(const GpsFix& fix) {
    std::lock_guard<std::mutex> lock(g_gps_mutex);
    g_latest_gps_fix = fix;
}

static GpsFix latest_gps_fix() {
    std::lock_guard<std::mutex> lock(g_gps_mutex);
    return g_latest_gps_fix;
}

static void apply_latest_gps_fix(EvidenceEvent& ev) {
    GpsFix fix = latest_gps_fix();
    if (!fix.valid) {
        return;
    }

    auto now = std::chrono::steady_clock::now();
    if (fix.received_at != std::chrono::steady_clock::time_point::min()) {
        auto age = std::chrono::duration_cast<std::chrono::seconds>(now - fix.received_at);
        if (age.count() > 15) {
            return;
        }
    }

    ev.gps_fix_valid = true;
    ev.gps_latitude = fix.latitude;
    ev.gps_longitude = fix.longitude;
    ev.gps_altitude_m = fix.altitude_m;
    ev.gps_speed_knots = fix.speed_knots;
    ev.gps_timestamp_utc = fix.timestamp_utc;
}

static bool parse_nmea_sentence(const std::string& line, GpsFix& fix) {
    if (line.empty() || line[0] != '$') {
        return false;
    }

    std::string payload = line.substr(1);
    std::string::size_type star_pos = payload.find('*');
    if (star_pos != std::string::npos) {
        payload = payload.substr(0, star_pos);
    }

    std::vector<std::string> fields = split_csv_fields(payload);
    if (fields.empty()) {
        return false;
    }

    const std::string& sentence_type = fields[0];
    GpsFix candidate = latest_gps_fix();
    bool updated = false;

    if ((sentence_type == "GPRMC" || sentence_type == "GNRMC") && fields.size() >= 10) {
        if (fields[2] != "A") {
            return false;
        }

        double latitude = 0.0;
        double longitude = 0.0;
        double speed_knots = 0.0;
        if (!parse_nmea_coordinate(fields[3], fields[4], true, latitude) ||
            !parse_nmea_coordinate(fields[5], fields[6], false, longitude)) {
            return false;
        }
        parse_double_value(fields[7], speed_knots);

        candidate.valid = true;
        candidate.latitude = latitude;
        candidate.longitude = longitude;
        candidate.speed_knots = speed_knots;
        candidate.timestamp_utc = nmea_date_time_to_utc(fields[1], fields[9]);
        candidate.received_at = std::chrono::steady_clock::now();
        updated = true;
    } else if ((sentence_type == "GPGGA" || sentence_type == "GNGGA") && fields.size() >= 10) {
        double latitude = 0.0;
        double longitude = 0.0;
        double altitude_m = 0.0;
        int fix_quality = std::atoi(fields[6].c_str());
        if (fix_quality <= 0) {
            return false;
        }
        if (!parse_nmea_coordinate(fields[2], fields[3], true, latitude) ||
            !parse_nmea_coordinate(fields[4], fields[5], false, longitude)) {
            return false;
        }
        parse_double_value(fields[9], altitude_m);

        candidate.valid = true;
        candidate.latitude = latitude;
        candidate.longitude = longitude;
        candidate.altitude_m = altitude_m;
        if (candidate.timestamp_utc.empty()) {
            candidate.timestamp_utc = today_with_nmea_time(fields[1]);
        }
        candidate.received_at = std::chrono::steady_clock::now();
        updated = true;
    } else if ((sentence_type == "GPGLL" || sentence_type == "GNGLL") && fields.size() >= 7) {
        if (fields[6] != "A") {
            return false;
        }

        double latitude = 0.0;
        double longitude = 0.0;
        if (!parse_nmea_coordinate(fields[1], fields[2], true, latitude) ||
            !parse_nmea_coordinate(fields[3], fields[4], false, longitude)) {
            return false;
        }

        candidate.valid = true;
        candidate.latitude = latitude;
        candidate.longitude = longitude;
        if (candidate.timestamp_utc.empty() && !fields[5].empty()) {
            candidate.timestamp_utc = today_with_nmea_time(fields[5]);
        }
        candidate.received_at = std::chrono::steady_clock::now();
        updated = true;
    }

    if (!updated) {
        return false;
    }

    if (candidate.timestamp_utc.empty()) {
        candidate.timestamp_utc = utc_now_iso8601();
    }
    fix = candidate;
    return true;
}

static int open_gps_serial_fd(const std::string& device_path, int baud_rate) {
    int fd = open(device_path.c_str(), O_RDONLY | O_NOCTTY);
    if (fd < 0) {
        return -1;
    }

    struct termios tty{};
    if (tcgetattr(fd, &tty) != 0) {
        close(fd);
        return -1;
    }

    cfmakeraw(&tty);
    speed_t baud = baud_to_termios(baud_rate);
    cfsetispeed(&tty, baud);
    cfsetospeed(&tty, baud);
    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~PARENB;
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;
    tty.c_cc[VMIN] = 0;
    tty.c_cc[VTIME] = 10;

    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
        close(fd);
        return -1;
    }

    return fd;
}

static int open_gps_tcp_fd(const std::string& host, int port) {
    struct addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    struct addrinfo* result = nullptr;
    const std::string port_text = std::to_string(port);
    if (getaddrinfo(host.c_str(), port_text.c_str(), &hints, &result) != 0) {
        return -1;
    }

    int fd = -1;
    for (struct addrinfo* item = result; item != nullptr; item = item->ai_next) {
        fd = socket(item->ai_family, item->ai_socktype, item->ai_protocol);
        if (fd < 0) {
            continue;
        }
        if (connect(fd, item->ai_addr, item->ai_addrlen) == 0) {
            break;
        }
        close(fd);
        fd = -1;
    }

    freeaddrinfo(result);
    return fd;
}

static void gps_reader_loop() {
    const std::string gps_host = trim_copy(getenv_string("ALPR_GPS_HOST")).empty()
        ? "192.168.1.1"
        : trim_copy(getenv_string("ALPR_GPS_HOST"));
    const int gps_port = getenv_int("ALPR_GPS_PORT", 11010);
    const std::string device_path = trim_copy(getenv_string("ALPR_GPS_DEVICE")).empty()
        ? ""
        : trim_copy(getenv_string("ALPR_GPS_DEVICE"));
    const int baud_rate = getenv_int("ALPR_GPS_BAUD", 4800);

    while (true) {
        int fd = open_gps_tcp_fd(gps_host, gps_port);
        if (fd < 0 && !device_path.empty()) {
            fd = open_gps_serial_fd(device_path, baud_rate);
        }
        if (fd < 0) {
            std::this_thread::sleep_for(std::chrono::seconds(2));
            continue;
        }

        std::string buffer;
        buffer.reserve(256);
        char chunk[128];

        while (true) {
            ssize_t read_count = read(fd, chunk, sizeof(chunk));
            if (read_count <= 0) {
                break;
            }

            for (ssize_t i = 0; i < read_count; ++i) {
                char ch = chunk[i];
                if (ch == '\r') {
                    continue;
                }
                if (ch == '\n') {
                    GpsFix fix;
                    if (parse_nmea_sentence(buffer, fix)) {
                        store_latest_gps_fix(fix);
                    }
                    buffer.clear();
                    continue;
                }
                if (buffer.size() < 512) {
                    buffer.push_back(ch);
                }
            }
        }

        close(fd);
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

static void ensure_gps_reader_started() {
    std::call_once(g_gps_thread_once, []() {
        std::thread reader(gps_reader_loop);
        reader.detach();
        g_gps_thread_started.store(true);
    });
}

static bool write_text_file_atomic(const std::string& path, const std::string& content) {
    std::string tmp_path = path + ".tmp";
    std::ofstream out(tmp_path, std::ios::trunc);
    if (!out) {
        return false;
    }
    out << content;
    out.close();
    if (!out) {
        std::remove(tmp_path.c_str());
        return false;
    }
    if (std::rename(tmp_path.c_str(), path.c_str()) != 0) {
        std::remove(tmp_path.c_str());
        return false;
    }
    return true;
}

static bool write_image_file_atomic(const std::string& path, const cv::Mat& image) {
    std::string::size_type dot_pos = path.find_last_of('.');
    std::string tmp_path;
    if (dot_pos == std::string::npos) {
        tmp_path = path + ".tmp.jpg";
    } else {
        tmp_path = path.substr(0, dot_pos) + ".tmp" + path.substr(dot_pos);
    }
    std::vector<int> params{cv::IMWRITE_JPEG_QUALITY, RUNTIME_PREVIEW_JPEG_QUALITY};
    if (!cv::imwrite(tmp_path, image, params)) {
        std::remove(tmp_path.c_str());
        return false;
    }
    if (std::rename(tmp_path.c_str(), path.c_str()) != 0) {
        std::remove(tmp_path.c_str());
        return false;
    }
    return true;
}

static LiveEndpoint parse_live_endpoint(const std::string& url) {
    LiveEndpoint endpoint;
    const std::string prefix = "http://";
    if (url.rfind(prefix, 0) != 0) {
        return endpoint;
    }

    std::string rest = url.substr(prefix.size());
    std::string::size_type path_pos = rest.find('/');
    std::string host_port = path_pos == std::string::npos ? rest : rest.substr(0, path_pos);
    endpoint.path = path_pos == std::string::npos ? "/api/live-event" : rest.substr(path_pos);
    if (endpoint.path.empty()) {
        endpoint.path = "/api/live-event";
    }

    std::string::size_type colon_pos = host_port.rfind(':');
    if (colon_pos == std::string::npos) {
        endpoint.host = host_port;
        endpoint.port = 80;
    } else {
        endpoint.host = host_port.substr(0, colon_pos);
        std::string port_str = host_port.substr(colon_pos + 1);
        if (port_str.empty()) {
            return endpoint;
        }
        endpoint.port = std::atoi(port_str.c_str());
    }

    if (endpoint.host.empty() || endpoint.port <= 0) {
        return endpoint;
    }

    endpoint.valid = true;
    return endpoint;
}

static LiveSourceInfo live_source_from_code(const std::string& code) {
    LiveSourceInfo source;
    std::string upper = uppercase_copy(trim_copy(code));
    if (upper == "LF") {
        source.code = "LF";
        source.label = "Left Front";
    } else if (upper == "RF") {
        source.code = "RF";
        source.label = "RF camera";
    } else if (upper == "LR") {
        source.code = "LR";
        source.label = "Left Rear";
    } else if (upper == "RR") {
        source.code = "RR";
        source.label = "Right Rear";
    }

    source.valid = !source.code.empty();
    return source;
}

static LiveSourceInfo map_live_source(const std::string& video_source) {
    LiveSourceInfo direct = live_source_from_code(video_source);
    if (direct.valid) {
        return direct;
    }

    std::string override_map = getenv_string("ALPR_LIVE_SOURCE_MAP");
    if (!override_map.empty()) {
        std::stringstream ss(override_map);
        std::string entry;
        while (std::getline(ss, entry, ',')) {
            std::string::size_type equals_pos = entry.find('=');
            if (equals_pos == std::string::npos) {
                continue;
            }
            std::string key = trim_copy(entry.substr(0, equals_pos));
            std::string value = trim_copy(entry.substr(equals_pos + 1));
            if (key == video_source) {
                LiveSourceInfo mapped = live_source_from_code(value);
                if (mapped.valid) {
                    return mapped;
                }
            }
        }
    }

    if (video_source == "source_0") return live_source_from_code("LF");
    if (video_source == "source_1") return live_source_from_code("RF");
    if (video_source == "source_2") return live_source_from_code("LR");
    if (video_source == "source_3") return live_source_from_code("RR");

    return LiveSourceInfo{};
}

static void warn_live_publish_config_once(const std::string& message) {
    std::lock_guard<std::mutex> lock(g_live_log_mutex);
    if (g_live_publish_config_warned) {
        return;
    }
    g_live_publish_config_warned = true;
    std::fprintf(stderr, "%s\n", message.c_str());
}

static bool send_http_post_json(const LiveEndpoint& endpoint, const std::string& payload) {
    struct addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    struct addrinfo* result = nullptr;
    std::string port_str = std::to_string(endpoint.port);
    if (getaddrinfo(endpoint.host.c_str(), port_str.c_str(), &hints, &result) != 0) {
        return false;
    }

    bool success = false;
    for (struct addrinfo* rp = result; rp != nullptr && !success; rp = rp->ai_next) {
        int fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (fd < 0) {
            continue;
        }

        int flags = fcntl(fd, F_GETFL, 0);
        if (flags < 0) {
            close(fd);
            continue;
        }

        if (fcntl(fd, F_SETFL, flags | O_NONBLOCK) != 0) {
            close(fd);
            continue;
        }

        int connect_result = connect(fd, rp->ai_addr, rp->ai_addrlen);
        if (connect_result != 0 && errno != EINPROGRESS) {
            close(fd);
            continue;
        }

        fd_set write_fds;
        FD_ZERO(&write_fds);
        FD_SET(fd, &write_fds);
        struct timeval connect_timeout{};
        connect_timeout.tv_sec = 0;
        connect_timeout.tv_usec = 300000;

        int select_result = select(fd + 1, nullptr, &write_fds, nullptr, &connect_timeout);
        if (select_result <= 0) {
            close(fd);
            continue;
        }

        int socket_error = 0;
        socklen_t socket_error_len = sizeof(socket_error);
        if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &socket_error, &socket_error_len) != 0 ||
            socket_error != 0) {
            close(fd);
            continue;
        }

        fcntl(fd, F_SETFL, flags);

        struct timeval io_timeout{};
        io_timeout.tv_sec = 0;
        io_timeout.tv_usec = 500000;
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &io_timeout, sizeof(io_timeout));
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &io_timeout, sizeof(io_timeout));

        std::ostringstream request;
        request << "POST " << endpoint.path << " HTTP/1.1\r\n"
                << "Host: " << endpoint.host << ':' << endpoint.port << "\r\n"
                << "Content-Type: application/json\r\n"
                << "Content-Length: " << payload.size() << "\r\n"
                << "Connection: close\r\n\r\n"
                << payload;

        std::string request_text = request.str();
        const char* data = request_text.c_str();
        size_t remaining = request_text.size();
        bool send_failed = false;
        while (remaining > 0) {
            ssize_t sent = send(fd, data, remaining, 0);
            if (sent <= 0) {
                send_failed = true;
                break;
            }
            data += sent;
            remaining -= static_cast<size_t>(sent);
        }

        if (!send_failed) {
            char response[256] = {0};
            ssize_t received = recv(fd, response, sizeof(response) - 1, 0);
            if (received > 0) {
                std::string status_line(response, static_cast<size_t>(received));
                success = status_line.find(" 200 ") != std::string::npos ||
                          status_line.find(" 201 ") != std::string::npos ||
                          status_line.find(" 202 ") != std::string::npos;
            }
        }

        close(fd);
    }

    freeaddrinfo(result);
    return success;
}

static std::string build_live_event_payload(const EvidenceEvent& ev,
                                            const std::string& case_id,
                                            const LiveSourceInfo& source) {
    auto case_relative_path = [&](const std::string& relative_path) {
        if (relative_path.empty()) {
            return std::string();
        }
        return case_id + "/" + relative_path;
    };

    std::ostringstream out;
    out << "{";
    out << "\"event_id\":\"" << json_escape(ev.event_id) << "\",";
    out << "\"case_id\":\"" << json_escape(case_id) << "\",";
    out << "\"plate\":\"" << json_escape(ev.plate) << "\",";
    out << "\"vehicle_make\":\"" << json_escape(ev.vehicle_make) << "\",";
    out << "\"vehicle_type\":\"" << json_escape(ev.vehicle_type) << "\",";
    out << "\"vehicle_color\":\"" << json_escape(ev.vehicle_color) << "\",";
    out << "\"status\":\"" << json_escape(ev.event_type) << "\",";
    out << "\"confidence\":" << ev.confidence << ',';
    out << "\"source\":\"" << json_escape(source.code) << "\",";
    out << "\"source_label\":\"" << json_escape(source.label) << "\",";
    out << "\"timestamp_utc\":\"" << json_escape(ev.timestamp_utc) << "\",";
    out << "\"frame_number\":" << ev.frame_number << ',';
    out << "\"track_id\":" << ev.track_id << ',';
    out << "\"track_id_valid\":" << (ev.track_id_valid ? "true" : "false") << ',';
    out << "\"gps_fix_valid\":" << (ev.gps_fix_valid ? "true" : "false") << ',';
    out << "\"gps_latitude\":" << json_number_or_null(ev.gps_fix_valid, ev.gps_latitude, 6) << ',';
    out << "\"gps_longitude\":" << json_number_or_null(ev.gps_fix_valid, ev.gps_longitude, 6) << ',';
    out << "\"gps_altitude_m\":" << json_number_or_null(ev.gps_fix_valid, ev.gps_altitude_m, 1) << ',';
    out << "\"gps_speed_knots\":" << json_number_or_null(ev.gps_fix_valid, ev.gps_speed_knots, 2) << ',';
    out << "\"gps_timestamp_utc\":" << json_string_or_null(ev.gps_fix_valid ? ev.gps_timestamp_utc : std::string()) << ',';
    out << "\"full_frame_path\":\"" << json_escape(case_relative_path(ev.full_frame_path)) << "\",";
    out << "\"annotated_frame_path\":\"" << json_escape(case_relative_path(ev.annotated_frame_path)) << "\",";
    out << "\"plate_crop_path\":\"" << json_escape(case_relative_path(ev.plate_crop_path)) << "\",";
    out << "\"vehicle_crop_path\":\"" << json_escape(case_relative_path(ev.vehicle_crop_path)) << "\",";
    out << "\"ca_pattern\":\"" << json_escape(ev.ca_pattern) << "\"";
    out << "}";
    return out.str();
}

static void publish_live_event_async(const EvidenceEvent& ev, const std::string& case_id) {
    if (ev.event_type != "CONFIRMED" && ev.event_type != "LOCKED") {
        return;
    }

    std::string endpoint_url = getenv_string("ALPR_LIVE_ENDPOINT");
    if (endpoint_url.empty()) {
        endpoint_url = "http://127.0.0.1:8080/api/live-event";
    }

    LiveEndpoint endpoint = parse_live_endpoint(endpoint_url);
    if (!endpoint.valid) {
        warn_live_publish_config_once("ALPR live publish disabled: invalid ALPR_LIVE_ENDPOINT (expected http://host:port/path)");
        return;
    }

    LiveSourceInfo source = map_live_source(ev.video_source);
    if (!source.valid) {
        warn_live_publish_config_once("ALPR live publish disabled: unmapped video source; set ALPR_LIVE_SOURCE_MAP, e.g. source_0=LF,source_1=RF");
        return;
    }

    std::string payload = build_live_event_payload(ev, case_id, source);
    std::thread([endpoint, payload]() {
        send_http_post_json(endpoint, payload);
    }).detach();
}

void update_runtime_source_status(const std::string& case_id,
                                  const std::string& video_source,
                                  int frame_number) {
    if (case_id.empty() || video_source.empty()) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_runtime_status_mutex);
    RuntimeSourceState& source = g_runtime_sources[video_source];
    source.source = video_source;
    source.last_seen_utc = utc_now_iso8601();
    source.last_frame_number = frame_number;

    auto now = std::chrono::steady_clock::now();
    if (g_runtime_status_last_flush != std::chrono::steady_clock::time_point::min()) {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - g_runtime_status_last_flush);
        if (elapsed.count() < 1000) {
            return;
        }
    }

    if (!ensure_dir("runtime")) {
        return;
    }

    std::ostringstream out;
    out << "{";
    out << "\"current_case_id\":\"" << json_escape(case_id) << "\",";
    out << "\"updated_utc\":\"" << json_escape(utc_now_iso8601()) << "\",";
    out << "\"sources\":[";

    bool first = true;
    for (const auto& entry : g_runtime_sources) {
        if (!first) {
            out << ',';
        }
        first = false;
        out << "{";
        out << "\"source\":\"" << json_escape(entry.second.source) << "\",";
        out << "\"last_seen_utc\":\"" << json_escape(entry.second.last_seen_utc) << "\",";
        out << "\"last_frame_number\":" << entry.second.last_frame_number;
        if (!entry.second.preview_path.empty()) {
            out << ",\"preview_path\":\"" << json_escape(entry.second.preview_path) << "\"";
        }
        if (!entry.second.preview_updated_utc.empty()) {
            out << ",\"preview_updated_utc\":\"" << json_escape(entry.second.preview_updated_utc) << "\"";
        }
        if (entry.second.preview_sequence > 0) {
            out << ",\"preview_sequence\":" << entry.second.preview_sequence;
        }
        if (entry.second.preview_overlay_width > 0 && entry.second.preview_overlay_height > 0) {
            out << ",\"preview_overlay_width\":" << entry.second.preview_overlay_width;
            out << ",\"preview_overlay_height\":" << entry.second.preview_overlay_height;
        }
        if (!entry.second.preview_detections.empty()) {
            out << ",\"preview_detections\":[";
            bool first_detection = true;
            for (const auto& detection : entry.second.preview_detections) {
                if (!first_detection) {
                    out << ',';
                }
                first_detection = false;
                out << "{";
                out << "\"left\":" << detection.left;
                out << ",\"top\":" << detection.top;
                out << ",\"width\":" << detection.width;
                out << ",\"height\":" << detection.height;
                out << ",\"confidence\":" << detection.confidence;
                out << ",\"plate\":\"" << json_escape(detection.plate) << "\"";
                out << ",\"focus_state\":\"" << json_escape(detection.focus_state) << "\"";
                out << "}";
            }
            out << "]";
        }
        out << "}";
    }

    out << "]}";

    if (write_text_file_atomic("runtime/alpr_status.json", out.str())) {
        g_runtime_status_last_flush = now;
    }
}

bool should_update_runtime_source_preview(const std::string& video_source) {
    if (video_source.empty()) {
        return false;
    }

    auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(g_runtime_status_mutex);
    RuntimeSourceState& source = g_runtime_sources[video_source];
    if (source.last_preview_write == std::chrono::steady_clock::time_point::min()) {
        return true;
    }

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - source.last_preview_write);
    return elapsed.count() >= RUNTIME_PREVIEW_INTERVAL_MS;
}

void update_runtime_source_preview(const cv::Mat& frame,
                                  const std::string& video_source,
                                  const std::vector<RuntimePreviewDetection>& detections) {
    if (frame.empty() || video_source.empty()) {
        return;
    }

    auto now = std::chrono::steady_clock::now();
    if (!should_update_runtime_source_preview(video_source)) {
        return;
    }

    if (!ensure_dir("runtime") || !ensure_dir("runtime/previews")) {
        return;
    }

    cv::Mat preview_frame;
    cv::Size preview_size = runtime_preview_size_for_frame(frame);
    cv::resize(frame, preview_frame, preview_size, 0.0, 0.0, cv::INTER_AREA);

    std::vector<RuntimePreviewDetection> scaled_detections;
    scaled_detections.reserve(detections.size());
    for (const auto& detection : detections) {
        RuntimePreviewDetection scaled = scale_preview_detection(
            detection, frame.cols, frame.rows, preview_frame.cols, preview_frame.rows);
        if (scaled.width <= 0 || scaled.height <= 0) {
            continue;
        }
        scaled_detections.push_back(scaled);
    }

    std::string relative_path = std::string("previews/") + video_source + ".jpg";
    std::string absolute_path = std::string("runtime/") + relative_path;
    if (!write_image_file_atomic(absolute_path, preview_frame)) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_runtime_status_mutex);
    RuntimeSourceState& source = g_runtime_sources[video_source];
    source.source = video_source;
    source.preview_path = relative_path;
    source.preview_updated_utc = utc_now_iso8601();
    source.preview_sequence++;
    source.preview_overlay_width = preview_frame.cols;
    source.preview_overlay_height = preview_frame.rows;
    source.preview_detections = scaled_detections;
    source.last_preview_write = now;
}

static std::string short_event_id(const std::string& event_id) {
    std::string::size_type pos = event_id.rfind("_evt_");
    if (pos == std::string::npos) {
        return event_id;
    }
    return event_id.substr(pos + 1);
}

static std::string display_track_id(bool track_id_valid, uint64_t track_id) {
    if (!track_id_valid) {
        return "N/A";
    }
    return std::to_string(track_id);
}

static cv::Scalar event_color(const std::string& event_type) {
    if (event_type == "LOCKED") {
        return cv::Scalar(0, 255, 0);
    }
    if (event_type == "CONFIRMED") {
        return cv::Scalar(0, 255, 255);
    }
    return cv::Scalar(0, 0, 255);
}

static bool is_debug_event(const std::string& event_type) {
    return event_type == "DEBUG";
}

struct FooterLine {
    std::string text;
    double font_scale = 0.5;
};

struct FooterStyle {
    double font_scale = 0.5;
    double min_font_scale = 0.38;
    int thickness = 1;
    int line_gap = 8;
    int top_pad = 12;
    int bottom_pad = 12;
    int left_pad = 12;
    int right_pad = 12;
};

static std::string compact_event_label(const std::string& event_id) {
    std::string short_id = short_event_id(event_id);
    if (short_id.rfind("evt_", 0) == 0) {
        return short_id.substr(4);
    }
    return short_id;
}

static std::string compact_timestamp(const std::string& timestamp_utc) {
    if (timestamp_utc.size() >= 20) {
        return timestamp_utc.substr(11);
    }
    return timestamp_utc;
}

static std::string json_number_or_null(bool valid, double value, int precision) {
    if (!valid) {
        return "null";
    }

    std::ostringstream out;
    out << std::fixed << std::setprecision(precision) << value;
    return out.str();
}

static std::string json_string_or_null(const std::string& value) {
    if (value.empty()) {
        return "null";
    }

    return std::string("\"") + json_escape(value) + "\"";
}

static std::string format_gps_summary(const EvidenceEvent& ev) {
    if (!ev.gps_fix_valid) {
        return "GPS: NO FIX";
    }

    std::ostringstream out;
    out << std::fixed << std::setprecision(6)
        << "GPS: " << ev.gps_latitude << ", " << ev.gps_longitude;
    if (ev.gps_altitude_m != 0.0) {
        out << "  ALT: " << std::setprecision(1) << ev.gps_altitude_m << "m";
    }
    if (!ev.gps_timestamp_utc.empty()) {
        out << "  FIX: " << compact_timestamp(ev.gps_timestamp_utc);
    }
    return out.str();
}

static FooterLine fit_footer_line(const std::string& text,
                                  const FooterStyle& style,
                                  int max_width) {
    FooterLine line{text, style.font_scale};
    int baseline = 0;

    while (line.font_scale > style.min_font_scale) {
        cv::Size size = cv::getTextSize(line.text, cv::FONT_HERSHEY_SIMPLEX,
                                        line.font_scale, style.thickness, &baseline);
        if (size.width <= max_width) {
            return line;
        }
        line.font_scale -= 0.02;
    }

    line.font_scale = style.min_font_scale;
    if (line.text.empty()) {
        return line;
    }

    std::string trimmed = line.text;
    while (trimmed.size() > 4) {
        trimmed.pop_back();
        std::string candidate = trimmed + "...";
        cv::Size size = cv::getTextSize(candidate, cv::FONT_HERSHEY_SIMPLEX,
                                        line.font_scale, style.thickness, &baseline);
        if (size.width <= max_width) {
            line.text = candidate;
            return line;
        }
    }

    line.text = trimmed;
    return line;
}

static std::vector<FooterLine> build_full_footer_lines(const std::string& case_id,
                                                       const std::string& event_id,
                                                       const std::string& plate,
                                                       int confidence,
                                                       bool track_id_valid,
                                                       uint64_t track_id,
                                                       const std::string& timestamp_utc,
                                                       const EvidenceEvent* ev = nullptr) {
    std::vector<FooterLine> lines;
    lines.push_back({"PLATE: " + plate + "    CONF: " + std::to_string(confidence) +
                     "    TRACK: " + display_track_id(track_id_valid, track_id), 0.5});
    lines.push_back({"CASE: " + case_id, 0.5});
    lines.push_back({"EVENT: " + event_id, 0.5});
    lines.push_back({"UTC: " + timestamp_utc, 0.5});
    if (ev != nullptr) {
        lines.push_back({format_gps_summary(*ev), 0.46});
    }
    return lines;
}

static std::vector<FooterLine> build_crop_footer_lines(const std::string& event_id,
                                                       const std::string& plate,
                                                       int confidence) {
    std::vector<FooterLine> lines;
    lines.push_back({"PLATE: " + plate, 0.42});
    lines.push_back({"CONF: " + std::to_string(confidence), 0.42});
    lines.push_back({"EVT: " + compact_event_label(event_id), 0.42});
    return lines;
}

static std::vector<FooterLine> build_vehicle_crop_footer_lines(const std::string& event_id,
                                                               const std::string& plate,
                                                               const std::string& vehicle_make,
                                                               const std::string& vehicle_type,
                                                               const std::string& vehicle_color) {
    std::vector<FooterLine> lines;
    lines.push_back({"PLATE: " + plate, 0.40});
    lines.push_back({"MAKE: " + (vehicle_make.empty() ? std::string("Unknown") : vehicle_make), 0.40});
    lines.push_back({"TYPE: " + (vehicle_type.empty() ? std::string("Unknown") : vehicle_type), 0.40});
    lines.push_back({"COLOR: " + (vehicle_color.empty() ? std::string("Unknown") : vehicle_color), 0.40});
    lines.push_back({"EVT: " + compact_event_label(event_id), 0.40});
    return lines;
}

static cv::Mat prepare_plate_crop_for_footer(const cv::Mat& crop) {
    if (crop.empty()) {
        return crop;
    }

    const int min_width = 220;
    if (crop.cols >= min_width) {
        return crop;
    }

    double scale = static_cast<double>(min_width) / static_cast<double>(crop.cols);
    cv::Mat resized;
    cv::resize(crop, resized,
               cv::Size(min_width, std::max(1, static_cast<int>(crop.rows * scale))),
               0.0, 0.0, cv::INTER_CUBIC);
    return resized;
}

static cv::Mat with_trace_footer(const cv::Mat& image,
                                 const std::vector<FooterLine>& raw_lines,
                                 const FooterStyle& style) {
    if (image.empty()) {
        return image;
    }

    int max_width = std::max(1, image.cols - style.left_pad - style.right_pad);
    std::vector<FooterLine> lines;
    lines.reserve(raw_lines.size());

    int total_text_height = 0;
    for (const FooterLine& raw_line : raw_lines) {
        FooterLine fitted = raw_line;
        fitted.font_scale = raw_line.font_scale > 0.0 ? raw_line.font_scale : style.font_scale;
        FooterStyle line_style = style;
        line_style.font_scale = fitted.font_scale;
        line_style.min_font_scale = std::min(style.min_font_scale, fitted.font_scale);
        fitted = fit_footer_line(fitted.text, line_style, max_width);

        int baseline = 0;
        cv::Size size = cv::getTextSize(fitted.text, cv::FONT_HERSHEY_SIMPLEX,
                                        fitted.font_scale, style.thickness, &baseline);
        total_text_height += size.height + baseline;
        lines.push_back(fitted);
    }

    int footer_height = style.top_pad + total_text_height +
                        std::max(0, static_cast<int>(lines.size()) - 1) * style.line_gap +
                        style.bottom_pad;
    cv::Mat stamped;
    cv::copyMakeBorder(image, stamped, 0, footer_height, 0, 0,
                       cv::BORDER_CONSTANT, cv::Scalar(18, 18, 18));

    int y = image.rows + style.top_pad;
    for (const FooterLine& line : lines) {
        int baseline = 0;
        cv::Size size = cv::getTextSize(line.text, cv::FONT_HERSHEY_SIMPLEX,
                                        line.font_scale, style.thickness, &baseline);
        y += size.height;
        cv::putText(stamped,
                    line.text,
                    cv::Point(style.left_pad, y),
                    cv::FONT_HERSHEY_SIMPLEX,
                    line.font_scale,
                    cv::Scalar(235, 235, 235),
                    style.thickness,
                    cv::LINE_AA);
        y += baseline + style.line_gap;
    }

    return stamped;
}

static cv::Rect clamp_rect(const cv::Rect& rect, const cv::Size& size) {
    return rect & cv::Rect(0, 0, size.width, size.height);
}

static cv::Mat render_review_frame(const cv::Mat& frame,
                                   const EvidenceEvent& ev,
                                   const std::string& case_id,
                                   const std::string& trace_event_id,
                                   bool include_inset) {
    cv::Mat rendered = frame.clone();
    cv::Scalar color = event_color(ev.event_type);

    cv::Rect vehicle_rect(ev.veh_left, ev.veh_top, ev.veh_width, ev.veh_height);
    cv::Rect plate_rect(ev.plate_left, ev.plate_top, ev.plate_width, ev.plate_height);
    vehicle_rect = clamp_rect(vehicle_rect, rendered.size());
    plate_rect = clamp_rect(plate_rect, rendered.size());

    if (vehicle_rect.width > 0 && vehicle_rect.height > 0) {
        cv::rectangle(rendered, vehicle_rect, color, 2);
    }
    if (plate_rect.width > 0 && plate_rect.height > 0) {
        cv::rectangle(rendered, plate_rect, color, 2);
    }

    cv::putText(rendered,
                ev.plate + " [" + std::to_string(ev.confidence) + "]",
                cv::Point(std::max(0, ev.plate_left), std::max(20, ev.plate_top - 8)),
                cv::FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv::LINE_AA);

    cv::putText(rendered,
                "track " + display_track_id(ev.track_id_valid, ev.track_id),
                cv::Point(std::max(0, ev.veh_left),
                          std::min(rendered.rows - 10,
                                   std::max(30, ev.veh_top + ev.veh_height + 24))),
                cv::FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv::LINE_AA);

    if (is_debug_event(ev.event_type)) {
        cv::putText(rendered,
                    "DEBUG ONLY - NOT EVIDENCE",
                    cv::Point(16, 30),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                    cv::LINE_AA);
    }

    if (include_inset && plate_rect.width > 0 && plate_rect.height > 0) {
        cv::Mat crop = frame(plate_rect).clone();
        int inset_width = std::min(std::max(plate_rect.width * 3, 140), frame.cols / 3);
        int inset_height = std::min(std::max(plate_rect.height * 3, 70), frame.rows / 4);
        if (inset_width > 0 && inset_height > 0) {
            cv::Mat inset;
            cv::resize(crop, inset, cv::Size(inset_width, inset_height), 0.0, 0.0, cv::INTER_CUBIC);
            cv::rectangle(inset, cv::Rect(0, 0, inset.cols, inset.rows), color, 2);

            int margin = 16;
            int inset_x = std::max(0, frame.cols - inset.cols - margin);
            int inset_y = margin;
            cv::Rect target(inset_x, inset_y, inset.cols, inset.rows);
            target = clamp_rect(target, rendered.size());
            if (target.width > 0 && target.height > 0) {
                inset(cv::Rect(0, 0, target.width, target.height)).copyTo(rendered(target));
                cv::rectangle(rendered, target, color, 2);
                cv::putText(rendered,
                            "plate inset",
                            cv::Point(target.x, std::min(rendered.rows - 10, target.y + target.height + 18)),
                            cv::FONT_HERSHEY_SIMPLEX,
                            0.55,
                            color,
                            2,
                            cv::LINE_AA);
            }
        }
    }

    FooterStyle footer_style;
    footer_style.font_scale = 0.5;
    footer_style.min_font_scale = 0.42;
    footer_style.line_gap = 8;
    footer_style.top_pad = 14;
    footer_style.bottom_pad = 14;

    return with_trace_footer(rendered,
                             build_full_footer_lines(case_id, trace_event_id, ev.plate,
                                                     ev.confidence, ev.track_id_valid,
                                                     ev.track_id, ev.timestamp_utc, &ev),
                             footer_style);
}

bool ensure_session_json(const std::string& evidence_root,
                         const std::string& video_source,
                         const std::string& model_version,
                         int reject_threshold,
                         int lock_threshold) {
    ensure_gps_reader_started();

    std::lock_guard<std::mutex> lock(g_session_mutex);
    if (g_session_written) return true;

    if (!ensure_dir(evidence_root)) return false;

    if (g_session_case_id.empty()) {
        std::string created = utc_now_iso8601();
        std::string date = created.substr(0, 10);
        std::replace(date.begin(), date.end(), '-', '_');

        ensure_dir("evidence");

        int sequence = 1;
        while (true) {
            std::ostringstream oss;
            oss << "case_" << date << "_" << std::setw(3) << std::setfill('0') << sequence;
            std::string candidate = oss.str();
            if (!path_exists("evidence/" + candidate)) {
                g_session_case_id = candidate;
                break;
            }
            ++sequence;
        }

        g_session_created_utc = created;
    }

    std::ofstream out(evidence_root + "/session.json", std::ios::trunc);
    if (!out) return false;

    out << "{\n"
        << "  \"case_id\": \"" << json_escape(g_session_case_id) << "\",\n"
        << "  \"video_source\": \"" << json_escape(video_source) << "\",\n"
        << "  \"model_version\": \"" << json_escape(model_version) << "\",\n"
        << "  \"created_utc\": \"" << json_escape(g_session_created_utc) << "\",\n"
        << "  \"gps\": {\n"
        << "    \"source\": \"mifi-tcp\",\n"
        << "    \"endpoint\": \""
        << json_escape(
               (trim_copy(getenv_string("ALPR_GPS_HOST")).empty() ? "192.168.1.1" : trim_copy(getenv_string("ALPR_GPS_HOST")))
               + ":" + std::to_string(getenv_int("ALPR_GPS_PORT", 11010))
           )
        << "\"\n"
        << "  }";

    if (reject_threshold >= 0 || lock_threshold >= 0) {
        out << ",\n  \"thresholds\": {\n";
        bool wrote_threshold = false;
        if (reject_threshold >= 0) {
            out << "    \"reject_threshold\": " << reject_threshold;
            wrote_threshold = true;
        }
        if (lock_threshold >= 0) {
            if (wrote_threshold) out << ",\n";
            out << "    \"lock_threshold\": " << lock_threshold;
        }
        out << "\n  }\n";
    } else {
        out << "\n";
    }

    out << "}\n";

    g_session_written = true;
    generate_case_review_index(evidence_root, std::string());
    return true;
}

bool append_event_jsonl(const std::string& jsonl_path, const EvidenceEvent& ev) {
    std::ofstream out(jsonl_path, std::ios::app);
    if (!out) return false;

    out << "{"
        << "\"event_id\":\"" << json_escape(ev.event_id) << "\","
        << "\"event_type\":\"" << json_escape(ev.event_type) << "\","
        << "\"plate\":\"" << json_escape(ev.plate) << "\","
        << "\"vehicle_make\":\"" << json_escape(ev.vehicle_make) << "\"," 
        << "\"vehicle_type\":\"" << json_escape(ev.vehicle_type) << "\"," 
        << "\"vehicle_color\":\"" << json_escape(ev.vehicle_color) << "\"," 
        << "\"confidence\":" << ev.confidence << ","
        << "\"frame_number\":" << ev.frame_number << ","
        << "\"track_id_valid\":" << (ev.track_id_valid ? "true" : "false") << ","
        << "\"track_id\":" << ev.track_id << ","
        << "\"video_source\":\"" << json_escape(ev.video_source) << "\","
        << "\"timestamp_utc\":\"" << json_escape(ev.timestamp_utc) << "\","
        << "\"gps_fix_valid\":" << (ev.gps_fix_valid ? "true" : "false") << ","
        << "\"gps_latitude\":" << json_number_or_null(ev.gps_fix_valid, ev.gps_latitude, 6) << ","
        << "\"gps_longitude\":" << json_number_or_null(ev.gps_fix_valid, ev.gps_longitude, 6) << ","
        << "\"gps_altitude_m\":" << json_number_or_null(ev.gps_fix_valid, ev.gps_altitude_m, 1) << ","
        << "\"gps_speed_knots\":" << json_number_or_null(ev.gps_fix_valid, ev.gps_speed_knots, 2) << ","
        << "\"gps_timestamp_utc\":" << json_string_or_null(ev.gps_fix_valid ? ev.gps_timestamp_utc : std::string()) << ","
        << "\"bbox_vehicle\":{"
            << "\"left\":" << ev.veh_left << ","
            << "\"top\":" << ev.veh_top << ","
            << "\"width\":" << ev.veh_width << ","
            << "\"height\":" << ev.veh_height
        << "},"
        << "\"bbox_plate\":{"
            << "\"left\":" << ev.plate_left << ","
            << "\"top\":" << ev.plate_top << ","
            << "\"width\":" << ev.plate_width << ","
            << "\"height\":" << ev.plate_height
        << "},"
        << "\"full_frame_path\":\"" << json_escape(ev.full_frame_path) << "\","
        << "\"plate_crop_path\":\"" << json_escape(ev.plate_crop_path) << "\","
        << "\"vehicle_crop_path\":\"" << json_escape(ev.vehicle_crop_path) << "\","
        << "\"annotated_frame_path\":\"" << json_escape(ev.annotated_frame_path) << "\"," 
        << "\"full_frame_sha256\":\"" << json_escape(ev.full_frame_sha256) << "\","
        << "\"plate_crop_sha256\":\"" << json_escape(ev.plate_crop_sha256) << "\","
        << "\"vehicle_crop_sha256\":\"" << json_escape(ev.vehicle_crop_sha256) << "\","
        << "\"annotated_frame_sha256\":\"" << json_escape(ev.annotated_frame_sha256) << "\"," 
        << "\"model_version\":\"" << json_escape(ev.model_version) << "\","
        << "\"notes\":\"" << json_escape(ev.notes) << "\","
        << "\"ca_pattern\":\"" << json_escape(ev.ca_pattern) << "\""
        << "}\n";

    std::string case_dir = dirname_from_path(jsonl_path);
    if (basename_from_path(case_dir) == "debug") {
        case_dir = dirname_from_path(case_dir);
    }
    generate_case_review_index(case_dir, std::string());
    return true;
}

bool save_event_images(const cv::Mat& frame,
                       int plate_left, int plate_top, int plate_width, int plate_height,
                       const std::string& case_id,
                       const std::string& event_id,
                       const std::string& full_frame_path,
                       const std::string& plate_crop_path) {
    if (frame.empty()) return false;

    FooterStyle full_style;
    cv::Mat traced_full = with_trace_footer(frame,
                                            build_full_footer_lines(case_id, event_id, "", 0,
                                                                    false, 0, ""),
                                            full_style);
    if (!cv::imwrite(full_frame_path, traced_full)) return false;

    cv::Rect plate_rect(plate_left, plate_top, plate_width, plate_height);
    cv::Rect bounds(0, 0, frame.cols, frame.rows);
    plate_rect = plate_rect & bounds;

    if (plate_rect.width <= 0 || plate_rect.height <= 0) return false;

    cv::Mat crop = prepare_plate_crop_for_footer(frame(plate_rect).clone());
    FooterStyle crop_style;
    crop_style.font_scale = 0.42;
    crop_style.min_font_scale = 0.34;
    crop_style.line_gap = 6;
    crop_style.top_pad = 10;
    crop_style.bottom_pad = 10;
    cv::Mat traced_crop = with_trace_footer(crop,
                                            build_crop_footer_lines(event_id, "", 0),
                                            crop_style);
    return cv::imwrite(plate_crop_path, traced_crop);
}

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
                          int plate_left, int plate_top, int plate_width, int plate_height,
                          const std::string& ca_pattern) {
    if (plate.size() < 5) return false;

    if (!ensure_session_json(evidence_root, video_source, model_version)) {
        return false;
    }

    std::string frames_dir = evidence_root + "/frames";
    if (!ensure_dir(frames_dir)) return false;

    EvidenceEvent ev;
    ev.event_id = next_event_id();
    ev.event_type = event_type;
    ev.plate = plate;
    ev.vehicle_make = vehicle_make;
    ev.vehicle_type = vehicle_type;
    ev.vehicle_color = vehicle_color;
    ev.confidence = confidence;
    ev.frame_number = frame_number;
    ev.track_id_valid = track_id_valid;
    ev.track_id = track_id;
    ev.video_source = video_source;
    ev.timestamp_utc = utc_now_iso8601();
    apply_latest_gps_fix(ev);
    ev.ca_pattern = ca_pattern;

    ev.veh_left = veh_left;
    ev.veh_top = veh_top;
    ev.veh_width = veh_width;
    ev.veh_height = veh_height;

    ev.plate_left = plate_left;
    ev.plate_top = plate_top;
    ev.plate_width = plate_width;
    ev.plate_height = plate_height;

    ev.model_version = model_version;
    ev.notes = "auto-generated by ALPR pipeline";

    std::string case_id = basename_from_path(evidence_root);
    std::string trace_event_id = short_event_id(ev.event_id);

    std::string prefix = ev.event_id + "_" +
                         std::string(event_type == "LOCKED" ? "locked" : "confirmed");
    ev.full_frame_path = "frames/" + prefix + "_full.jpg";
    ev.plate_crop_path = "frames/" + prefix + "_plate.jpg";
    ev.vehicle_crop_path = "frames/" + prefix + "_vehicle.jpg";
    if (event_type == "LOCKED") {
        ev.annotated_frame_path = "frames/" + prefix + "_annotated.jpg";
    }

    if (frame.empty()) {
        ev.full_frame_path.clear();
        ev.plate_crop_path.clear();
        ev.vehicle_crop_path.clear();
        ev.annotated_frame_path.clear();
        ev.notes = "auto-generated by ALPR pipeline; image capture unavailable (empty frame)";
        bool json_ok = append_event_jsonl(evidence_root + "/events.jsonl", ev);
        if (json_ok) {
            publish_live_event_async(ev, case_id);
        }
        return json_ok;
    }

    std::string full_frame_abs = evidence_root + "/" + ev.full_frame_path;
    std::string plate_crop_abs = evidence_root + "/" + ev.plate_crop_path;
    std::string vehicle_crop_abs = evidence_root + "/" + ev.vehicle_crop_path;

    cv::Rect plate_rect(plate_left, plate_top, plate_width, plate_height);
    cv::Rect vehicle_rect(veh_left, veh_top, veh_width, veh_height);
    cv::Rect bounds(0, 0, frame.cols, frame.rows);
    plate_rect = plate_rect & bounds;
    vehicle_rect = vehicle_rect & bounds;
    if (plate_rect.width <= 0 || plate_rect.height <= 0) {
        return false;
    }

    cv::Mat review_full = render_review_frame(frame, ev, case_id, trace_event_id, true);
    if (!cv::imwrite(full_frame_abs, review_full)) {
        return false;
    }

    cv::Mat crop = prepare_plate_crop_for_footer(frame(plate_rect).clone());
    FooterStyle crop_style;
    crop_style.font_scale = 0.42;
    crop_style.min_font_scale = 0.34;
    crop_style.line_gap = 6;
    crop_style.top_pad = 10;
    crop_style.bottom_pad = 10;
    cv::Mat traced_crop = with_trace_footer(crop,
                                            build_crop_footer_lines(trace_event_id, ev.plate,
                                                                    ev.confidence),
                                            crop_style);
    if (!cv::imwrite(plate_crop_abs, traced_crop)) {
        return false;
    }

    ev.full_frame_sha256 = sha256_file(full_frame_abs);
    ev.plate_crop_sha256 = sha256_file(plate_crop_abs);

    if (vehicle_rect.width > 0 && vehicle_rect.height > 0) {
        cv::Mat vehicle_crop = frame(vehicle_rect).clone();
        FooterStyle vehicle_style;
        vehicle_style.font_scale = 0.40;
        vehicle_style.min_font_scale = 0.32;
        vehicle_style.line_gap = 6;
        vehicle_style.top_pad = 10;
        vehicle_style.bottom_pad = 10;
        cv::Mat traced_vehicle = with_trace_footer(
            vehicle_crop,
            build_vehicle_crop_footer_lines(trace_event_id, ev.plate,
                                            ev.vehicle_make, ev.vehicle_type, ev.vehicle_color),
            vehicle_style);
        if (cv::imwrite(vehicle_crop_abs, traced_vehicle)) {
            ev.vehicle_crop_sha256 = sha256_file(vehicle_crop_abs);
        } else {
            ev.vehicle_crop_path.clear();
        }
    } else {
        ev.vehicle_crop_path.clear();
    }

    if (event_type == "LOCKED" && !ev.annotated_frame_path.empty()) {
        cv::Mat annotated = render_review_frame(frame, ev, case_id, trace_event_id, true);

        std::string annotated_abs = evidence_root + "/" + ev.annotated_frame_path;
        if (!cv::imwrite(annotated_abs, annotated)) {
            return false;
        }
        ev.annotated_frame_sha256 = sha256_file(annotated_abs);
    }

    bool json_ok = append_event_jsonl(evidence_root + "/events.jsonl", ev);
    if (json_ok) {
        publish_live_event_async(ev, case_id);
    }
    return json_ok;
}

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
                       int plate_left, int plate_top, int plate_width, int plate_height,
                       const std::string& ca_pattern) {
    if (plate.size() < 5) return false;

    if (!ensure_session_json(evidence_root, video_source, model_version)) {
        return false;
    }

    std::string debug_root = evidence_root + "/debug";
    std::string debug_frames_dir = debug_root + "/frames";
    if (!ensure_dir(debug_frames_dir)) return false;

    EvidenceEvent ev;
    ev.event_id = next_event_id();
    ev.event_type = "DEBUG";
    ev.plate = plate;
    ev.vehicle_make = vehicle_make;
    ev.vehicle_type = vehicle_type;
    ev.vehicle_color = vehicle_color;
    ev.confidence = confidence;
    ev.frame_number = frame_number;
    ev.track_id_valid = track_id_valid;
    ev.track_id = track_id;
    ev.video_source = video_source;
    ev.timestamp_utc = utc_now_iso8601();
    apply_latest_gps_fix(ev);
    ev.ca_pattern = ca_pattern;
    ev.veh_left = veh_left;
    ev.veh_top = veh_top;
    ev.veh_width = veh_width;
    ev.veh_height = veh_height;
    ev.plate_left = plate_left;
    ev.plate_top = plate_top;
    ev.plate_width = plate_width;
    ev.plate_height = plate_height;
    ev.model_version = model_version;
    ev.notes = "DEBUG ONLY - NOT EVIDENCE; near-threshold diagnostic event";

    std::string case_id = basename_from_path(evidence_root);
    std::string trace_event_id = short_event_id(ev.event_id);
    std::string prefix = ev.event_id + "_debug";
    ev.full_frame_path = "debug/frames/" + prefix + "_full.jpg";
    ev.plate_crop_path = "debug/frames/" + prefix + "_plate.jpg";
    ev.vehicle_crop_path = "debug/frames/" + prefix + "_vehicle.jpg";

    if (frame.empty()) {
        ev.full_frame_path.clear();
        ev.plate_crop_path.clear();
        ev.vehicle_crop_path.clear();
        ev.notes = "DEBUG ONLY - NOT EVIDENCE; image capture unavailable (empty frame)";
        return append_event_jsonl(debug_root + "/debug.jsonl", ev);
    }

    cv::Rect plate_rect(plate_left, plate_top, plate_width, plate_height);
    cv::Rect vehicle_rect(veh_left, veh_top, veh_width, veh_height);
    cv::Rect bounds(0, 0, frame.cols, frame.rows);
    plate_rect = plate_rect & bounds;
    vehicle_rect = vehicle_rect & bounds;
    if (plate_rect.width <= 0 || plate_rect.height <= 0) {
        return false;
    }

    std::string full_frame_abs = evidence_root + "/" + ev.full_frame_path;
    std::string plate_crop_abs = evidence_root + "/" + ev.plate_crop_path;
    std::string vehicle_crop_abs = evidence_root + "/" + ev.vehicle_crop_path;

    cv::Mat review_full = render_review_frame(frame, ev, case_id, trace_event_id, true);
    if (!cv::imwrite(full_frame_abs, review_full)) {
        return false;
    }

    cv::Mat crop = prepare_plate_crop_for_footer(frame(plate_rect).clone());
    FooterStyle crop_style;
    crop_style.font_scale = 0.42;
    crop_style.min_font_scale = 0.34;
    crop_style.line_gap = 6;
    crop_style.top_pad = 10;
    crop_style.bottom_pad = 10;
    cv::Mat traced_crop = with_trace_footer(crop,
                                            build_crop_footer_lines(trace_event_id, ev.plate,
                                                                    ev.confidence),
                                            crop_style);
    if (!cv::imwrite(plate_crop_abs, traced_crop)) {
        return false;
    }

    ev.full_frame_sha256 = sha256_file(full_frame_abs);
    ev.plate_crop_sha256 = sha256_file(plate_crop_abs);

    if (vehicle_rect.width > 0 && vehicle_rect.height > 0) {
        cv::Mat vehicle_crop = frame(vehicle_rect).clone();
        FooterStyle vehicle_style;
        vehicle_style.font_scale = 0.40;
        vehicle_style.min_font_scale = 0.32;
        vehicle_style.line_gap = 6;
        vehicle_style.top_pad = 10;
        vehicle_style.bottom_pad = 10;
        cv::Mat traced_vehicle = with_trace_footer(
            vehicle_crop,
            build_vehicle_crop_footer_lines(trace_event_id, ev.plate,
                                            ev.vehicle_make, ev.vehicle_type, ev.vehicle_color),
            vehicle_style);
        if (cv::imwrite(vehicle_crop_abs, traced_vehicle)) {
            ev.vehicle_crop_sha256 = sha256_file(vehicle_crop_abs);
        } else {
            ev.vehicle_crop_path.clear();
        }
    } else {
        ev.vehicle_crop_path.clear();
    }

    return append_event_jsonl(debug_root + "/debug.jsonl", ev);
}
