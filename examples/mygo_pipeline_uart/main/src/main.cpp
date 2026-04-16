#include "maix_basic.hpp"
#include "maix_camera.hpp"
#include "maix_display.hpp"
#include "maix_image_cv.hpp"
#include "maix_jpg_stream.hpp"
#include "maix_key.hpp"
#include "main.h"

#include "TargetTracking/TargetTrackingPipeline.hpp"

#include <algorithm>
#include <arpa/inet.h>
#include <cerrno>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <fcntl.h>
#include <netinet/in.h>
#include <optional>
#include <sstream>
#include <string>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

using namespace maix;

namespace {

constexpr uint8_t kProtocolVersion = 0x01;
constexpr uint8_t kFlagIsResp = 0x80;
constexpr uint8_t kFlagRespOk = 0x40;

constexpr uint8_t CMD_SET_REPORT = 0xF8;
constexpr uint8_t CMD_APP_LIST = 0xF9;
constexpr uint8_t CMD_START_APP = 0xFA;
constexpr uint8_t CMD_EXIT_APP = 0xFB;
constexpr uint8_t CMD_CUR_APP_INFO = 0xFC;
constexpr uint8_t CMD_APP_INFO = 0xFD;

constexpr uint8_t APP_CMD_VISION_START = 0x10;
constexpr uint8_t APP_CMD_VISION_STOP = 0x11;
constexpr uint8_t APP_CMD_VISION_STATUS = 0x12;

constexpr uint32_t kHeader = 0xBBACCAAA;
constexpr const char *kAppId = "mygo_pipeline_uart";
constexpr const char *kAppName = "MyGo Pipeline TCP";
constexpr const char *kAppDesc = "Target tracking pipeline with TCP control/status";
constexpr float kHitThresholdPx = 30.0f;

enum class VisionAppState {
    Idle,
    Running,
    Stopped,
};

struct ProtocolFrame {
    uint8_t flags{0};
    uint8_t cmd{0};
    std::vector<uint8_t> body;
};

struct VisionStatusSnapshot {
    VisionAppState app_state{VisionAppState::Idle};
    std::string tracking_state{"Stopped"};
    bool active{false};
    bool tracking_enabled{false};
    bool can_scan{false};
    bool target_found{false};
    bool laser_found{false};
    int target_x{-1};
    int target_y{-1};
    std::string command;
};

struct PendingControl {
    bool recognition_start_requested{false};
    bool tracking_start_requested{false};
    bool stop_requested{false};
    bool tracking_init_pose_valid{false};
    int init_yaw_pwm{1500};
    int init_pitch_pwm{1500};
};

float pwm_to_angle_deg(int pwm, float zero_angle_deg)
{
    constexpr float kPwmPerDeg = 2000.0f / 270.0f;
    const float delta_deg = (static_cast<float>(pwm) - 1500.0f) / kPwmPerDeg;
    return std::clamp(zero_angle_deg + delta_deg, 0.0f, 270.0f);
}

const char *kStreamHtml = R"HTML(
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>MyGo Pipeline Stream</title>
</head>
<body>
    <h1>MyGo Pipeline Stream</h1>
    <p>Low-rate background JPEG stream from MaixCam2.</p>
    <img src="/stream" alt="Stream">
</body>
</html>
)HTML";

float clamp_value(float value, float low, float high)
{
    return std::max(low, std::min(value, high));
}

const char *state_to_text(TrackState state)
{
    switch (state) {
        case TrackState::Waiting: return "Waiting";
        case TrackState::Searching: return "Searching";
        case TrackState::Locked: return "Locked";
        case TrackState::Tracking: return "Tracking";
        default: return "Unknown";
    }
}

const char *vision_app_state_to_text(VisionAppState state)
{
    switch (state) {
        case VisionAppState::Idle: return "IDLE";
        case VisionAppState::Running: return "RUNNING";
        case VisionAppState::Stopped: return "STOPPED";
        default: return "UNKNOWN";
    }
}

std::string preview_text(const std::string &text, size_t max_len)
{
    std::string cleaned;
    cleaned.reserve(text.size());
    for (unsigned char ch : text) {
        if (ch == '\r' || ch == '\n' || ch == '\t') {
            cleaned.push_back(' ');
            continue;
        }

        if (std::isprint(ch)) {
            cleaned.push_back(static_cast<char>(ch));
        }
    }

    if (cleaned.empty()) {
        return "-";
    }

    if (cleaned.size() <= max_len) {
        return cleaned;
    }

    if (max_len <= 3) {
        return cleaned.substr(0, max_len);
    }
    return cleaned.substr(0, max_len - 3) + "...";
}

void append_u32_le(std::vector<uint8_t> &out, uint32_t value)
{
    out.push_back(static_cast<uint8_t>(value & 0xFF));
    out.push_back(static_cast<uint8_t>((value >> 8) & 0xFF));
    out.push_back(static_cast<uint8_t>((value >> 16) & 0xFF));
    out.push_back(static_cast<uint8_t>((value >> 24) & 0xFF));
}

uint32_t read_u32_le(const uint8_t *ptr)
{
    return static_cast<uint32_t>(ptr[0]) |
           (static_cast<uint32_t>(ptr[1]) << 8) |
           (static_cast<uint32_t>(ptr[2]) << 16) |
           (static_cast<uint32_t>(ptr[3]) << 24);
}

uint16_t read_u16_le(const uint8_t *ptr)
{
    return static_cast<uint16_t>(ptr[0]) | (static_cast<uint16_t>(ptr[1]) << 8);
}

uint16_t crc16_ibm(const uint8_t *data, size_t size)
{
    uint16_t crc = 0x0000;
    for (size_t i = 0; i < size; ++i) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit) {
            if (crc & 0x01) {
                crc = static_cast<uint16_t>((crc >> 1) ^ 0xA001);
            } else {
                crc = static_cast<uint16_t>(crc >> 1);
            }
        }
    }
    return crc;
}

void append_cstr(std::vector<uint8_t> &body, const std::string &text)
{
    body.insert(body.end(), text.begin(), text.end());
    body.push_back('\0');
}

std::vector<std::string> split_strings_with_nul(const std::vector<uint8_t> &data, size_t start)
{
    std::vector<std::string> result;
    std::string current;
    for (size_t i = start; i < data.size(); ++i) {
        if (data[i] == '\0') {
            result.push_back(current);
            current.clear();
        } else {
            current.push_back(static_cast<char>(data[i]));
        }
    }
    if (!current.empty()) {
        result.push_back(current);
    }
    return result;
}

void draw_pipeline_overlay(
    image::Image &img,
    const PipelineOutput &out,
    VisionAppState app_state,
    bool task_active,
    bool tcp_listening,
    bool tcp_client_connected,
    int tcp_port)
{
    const image::Color target_color = image::Color::from_rgb(255, 64, 64);
    const image::Color laser_color = image::Color::from_rgb(255, 255, 0);
    const image::Color ok_color = image::Color::from_rgb(0, 220, 120);
    const image::Color bad_color = image::Color::from_rgb(255, 72, 72);
    const image::Color warn_color = image::Color::from_rgb(255, 196, 0);
    const image::Color info_color = image::Color::from_rgb(0, 200, 255);
    const image::Color panel_bg = image::Color::from_rgb(18, 24, 34);
    const image::Color panel_border = image::Color::from_rgb(230, 230, 230);
    const image::Color panel_text = image::Color::from_rgb(245, 245, 245);
    const image::Color dark_good_bg = image::Color::from_rgb(8, 60, 32);
    const image::Color dark_warn_bg = image::Color::from_rgb(76, 50, 0);
    const image::Color dark_bad_bg = image::Color::from_rgb(76, 12, 12);
    const image::Color dark_info_bg = image::Color::from_rgb(10, 38, 76);
    const image::Color hit_color = ok_color;
    const image::Color miss_color = warn_color;
    const image::Color pred_color = image::Color::from_rgb(255, 96, 255);

    const int width = img.width();
    const int height = img.height();
    const int margin = std::max(10, width / 64);
    const int gap = margin;
    const int top_h = std::max(96, height / 5);
    const int banner_h = std::max(74, height / 7);
    const int bottom_h = std::max(46, height / 10);
    const int box_w = (width - margin * 2 - gap) / 2;
    const int left_x = margin;
    const int right_x = margin + box_w + gap;
    const int top_y = margin;
    const int banner_y = top_y + top_h + gap;
    const int info_y = banner_y + banner_h + gap;
    const int info_h = std::max(170, height - info_y - bottom_h - margin * 2);
    const int bottom_y = height - bottom_h - margin;
    const bool detect_hit = task_active && out.target_found && out.laser_found &&
                            out.laser_target_error_px < kHitThresholdPx;
    const int info_first_y = info_y + 46;
    const int info_preview_y = info_y + info_h - 24;
    const int info_step = std::max(18, (info_preview_y - info_first_y - 4) / 6);

    auto draw_panel = [&](int x, int y, int w, int h, const image::Color &fill, const image::Color &border) {
        (void)fill;
        img.draw_rect(x, y, w, h, border, 3);
    };

    auto draw_info_line = [&](int x,
                              int y,
                              const std::string &label,
                              const std::string &value,
                              const image::Color &value_color) {
        img.draw_string(x, y, label, panel_text, 0.95f, 2);
        img.draw_string(x + 114, y, value, value_color, 0.95f, 2);
    };

    const std::string tcp_main = !tcp_listening ? "LISTEN FAIL"
        : (tcp_client_connected ? "HOST CONNECTED" : "WAIT HOST");
    const std::string tcp_sub = std::string("PORT ") + std::to_string(tcp_port) + "  |  OK EXIT";
    const image::Color tcp_color = !tcp_listening ? bad_color
        : (tcp_client_connected ? ok_color : warn_color);
    const image::Color tcp_bg = !tcp_listening ? dark_bad_bg
        : (tcp_client_connected ? dark_good_bg : dark_warn_bg);

    const std::string track_text = task_active ? state_to_text(out.state) : "WAIT START";
    const std::string task_main = task_active ? "RUNNING"
        : (app_state == VisionAppState::Stopped ? "STOPPED" : "IDLE");
    const std::string task_sub = std::string("TRACK ") + track_text;
    const image::Color task_color = task_active ? ok_color
        : (app_state == VisionAppState::Stopped ? warn_color : info_color);
    const image::Color task_bg = task_active ? dark_good_bg
        : (app_state == VisionAppState::Stopped ? dark_warn_bg : dark_info_bg);

    std::string banner_text = "READY FOR START";
    std::string banner_sub = "Host A starts recognition, host B stops.";
    image::Color banner_color = info_color;
    image::Color banner_bg = dark_info_bg;
    if (!tcp_listening) {
        banner_text = "TCP SERVER ERROR";
        banner_sub = "Port bind failed. Restart app after fixing network.";
        banner_color = bad_color;
        banner_bg = dark_bad_bg;
    } else if (!tcp_client_connected) {
        banner_text = "WAIT HOST TCP";
        banner_sub = "Upper computer not connected yet.";
        banner_color = warn_color;
        banner_bg = dark_warn_bg;
    } else if (!task_active && app_state == VisionAppState::Stopped) {
        banner_text = "TASK STOPPED";
        banner_sub = "Recognition is closed. Waiting host start command.";
        banner_color = warn_color;
        banner_bg = dark_warn_bg;
    } else if (task_active && out.state == TrackState::Tracking) {
        banner_text = "TRACKING TARGET";
        banner_sub = out.target_found ? "Servo command is being generated." : "Tracking active.";
        banner_color = ok_color;
        banner_bg = dark_good_bg;
    } else if (task_active && out.state == TrackState::Locked) {
        banner_text = "LOCKED READY";
        banner_sub = "Locked target, preparing stable tracking.";
        banner_color = warn_color;
        banner_bg = dark_warn_bg;
    } else if (task_active && out.state == TrackState::Searching) {
        banner_text = "SEARCHING TARGET";
        banner_sub = "Pipeline running and waiting for a valid target.";
        banner_color = info_color;
        banner_bg = dark_info_bg;
    } else if (task_active) {
        banner_text = "PIPELINE ACTIVE";
        banner_sub = "Recognition is running.";
        banner_color = info_color;
        banner_bg = dark_info_bg;
    }

    if (out.target_found && out.target_pos.x >= 0.0f && out.target_pos.y >= 0.0f) {
        img.draw_rect(static_cast<int>(out.target_pos.x) - 3,
                      static_cast<int>(out.target_pos.y) - 3,
                      7,
                      7,
                      target_color,
                      2);
        img.draw_string(static_cast<int>(out.target_pos.x) + 6,
                        static_cast<int>(out.target_pos.y) - 10,
                        "target",
                        target_color,
                        1.2f);
    }

    if (out.roi_active && out.roi_rect.width > 0 && out.roi_rect.height > 0) {
        img.draw_rect(out.roi_rect.x,
                      out.roi_rect.y,
                      out.roi_rect.width,
                      out.roi_rect.height,
                      image::Color::from_rgb(64, 160, 255),
                      1);
    }

    if (out.laser_found && out.laser_pos.x >= 0.0f && out.laser_pos.y >= 0.0f) {
        img.draw_rect(static_cast<int>(out.laser_pos.x) - 3,
                      static_cast<int>(out.laser_pos.y) - 3,
                      7,
                      7,
                      laser_color,
                      2);
        img.draw_string(static_cast<int>(out.laser_pos.x) + 6,
                        static_cast<int>(out.laser_pos.y) - 10,
                        "laser",
                        laser_color,
                        1.2f);
    }

    if (task_active && out.target_found && out.laser_found) {
        img.draw_line(static_cast<int>(out.target_pos.x),
                      static_cast<int>(out.target_pos.y),
                      static_cast<int>(out.laser_pos.x),
                      static_cast<int>(out.laser_pos.y),
                      detect_hit ? hit_color : miss_color,
                      2);
    }

    if (out.predicted_pos_valid && out.predicted_pos.x >= 0.0f && out.predicted_pos.y >= 0.0f) {
        const int px = static_cast<int>(out.predicted_pos.x);
        const int py = static_cast<int>(out.predicted_pos.y);
        img.draw_line(px - 7, py - 7, px + 7, py + 7, pred_color, 2);
        img.draw_line(px - 7, py + 7, px + 7, py - 7, pred_color, 2);
        img.draw_string(px + 8, py - 10, "pred", pred_color, 1.0f);

        if (out.target_found && out.target_pos.x >= 0.0f && out.target_pos.y >= 0.0f) {
            img.draw_line(px,
                          py,
                          static_cast<int>(out.target_pos.x),
                          static_cast<int>(out.target_pos.y),
                          pred_color,
                          1);
        }
    }

    if (out.aim_pos.x >= 0.0f && out.aim_pos.y >= 0.0f) {
        const int aim_x = static_cast<int>(out.aim_pos.x);
        const int aim_y = static_cast<int>(out.aim_pos.y);
        img.draw_line(aim_x - 8, aim_y, aim_x + 8, aim_y, image::Color::from_rgb(0, 200, 255), 1);
        img.draw_line(aim_x, aim_y - 8, aim_x, aim_y + 8, image::Color::from_rgb(0, 200, 255), 1);
        img.draw_string(aim_x + 10,
                        aim_y - 10,
                        out.aim_from_laser ? "aim:laser" : "aim:center",
                        image::Color::from_rgb(0, 200, 255),
                        1.0f);
    }

    draw_panel(left_x, top_y, box_w, top_h, tcp_bg, tcp_color);
    img.draw_string(left_x + 14, top_y + 18, "TCP STATUS", panel_text, 0.95f, 2);
    img.draw_string(left_x + 14, top_y + 48, tcp_main, tcp_color, 1.35f, 2);
    img.draw_string(left_x + 14, top_y + 78, tcp_sub, panel_text, 0.88f, 2);

    draw_panel(right_x, top_y, box_w, top_h, task_bg, task_color);
    img.draw_string(right_x + 14, top_y + 18, "VISION TASK", panel_text, 0.95f, 2);
    img.draw_string(right_x + 14, top_y + 48, task_main, task_color, 1.35f, 2);
    img.draw_string(right_x + 14, top_y + 78, task_sub, panel_text, 0.88f, 2);

    draw_panel(left_x, banner_y, width - margin * 2, banner_h, banner_bg, banner_color);
    img.draw_string(left_x + 16, banner_y + 18, banner_text, banner_color, 1.65f, 2);
    img.draw_string(left_x + 16, banner_y + 52, banner_sub, panel_text, 0.92f, 2);

    draw_panel(left_x, info_y, box_w, info_h, panel_bg, panel_border);
    img.draw_string(left_x + 14, info_y + 16, "SYSTEM", panel_text, 0.95f, 2);
    draw_info_line(left_x + 14, info_first_y + info_step * 0, "APP", vision_app_state_to_text(app_state), task_color);
    draw_info_line(left_x + 14, info_first_y + info_step * 1, "TASK", task_active ? "ACTIVE" : "WAIT HOST", task_active ? ok_color : warn_color);
    draw_info_line(left_x + 14, info_first_y + info_step * 2, "TRACK", track_text, task_active ? info_color : warn_color);
    draw_info_line(left_x + 14, info_first_y + info_step * 3, "HOST", tcp_client_connected ? "ONLINE" : "OFFLINE", tcp_client_connected ? ok_color : bad_color);
    draw_info_line(left_x + 14, info_first_y + info_step * 4, "OUT", "TCP->ROS", info_color);
    draw_info_line(left_x + 14, info_first_y + info_step * 5, "EXIT", "OK BUTTON", warn_color);
    img.draw_string(left_x + 14,
                    info_preview_y,
                    std::string("LINK: ") + preview_text("tcp-control-only", 28),
                    panel_text,
                    0.82f,
                    2);

    draw_panel(right_x, info_y, box_w, info_h, panel_bg, panel_border);
    img.draw_string(right_x + 14, info_y + 16, "VISION", panel_text, 0.95f, 2);
    draw_info_line(right_x + 14, info_first_y + info_step * 0, "TARGET",
                   task_active ? (out.target_found ? "FOUND" : "MISS") : "N/A",
                   task_active ? (out.target_found ? ok_color : bad_color) : warn_color);
    draw_info_line(right_x + 14, info_first_y + info_step * 1, "LASER",
                   task_active ? (out.laser_found ? "FOUND" : "MISS") : "N/A",
                   task_active ? (out.laser_found ? ok_color : bad_color) : warn_color);
    draw_info_line(right_x + 14, info_first_y + info_step * 2, "HIT",
                   task_active && out.target_found && out.laser_found ? (detect_hit ? "HIT" : "MISS") : "N/A",
                   task_active && out.target_found && out.laser_found ? (detect_hit ? ok_color : warn_color) : warn_color);
    draw_info_line(right_x + 14, info_first_y + info_step * 3, "ERROR",
                   task_active && out.target_found && out.laser_found
                       ? preview_text(std::to_string(static_cast<int>(out.laser_target_error_px)) + " px", 10)
                       : "N/A",
                   task_active && out.target_found && out.laser_found ? info_color : warn_color);
    draw_info_line(right_x + 14, info_first_y + info_step * 4, "AIM",
                   task_active ? (out.aim_from_laser ? "LASER" : "CENTER") : "IDLE",
                   task_active ? info_color : warn_color);
    draw_info_line(right_x + 14, info_first_y + info_step * 5, "CMD",
                   task_active ? (!out.command.empty() ? "READY" : "EMPTY") : "IDLE",
                   task_active ? (!out.command.empty() ? ok_color : warn_color) : warn_color);
    img.draw_string(right_x + 14,
                    info_preview_y,
                    std::string("FRAME: ") + preview_text(out.command, 28),
                    panel_text,
                    0.82f,
                    2);

    draw_panel(left_x, bottom_y, width - margin * 2, bottom_h, panel_bg, panel_border);
    img.draw_string(left_x + 16,
                    bottom_y + 16,
                    "HOST A START   HOST B STOP   LOCAL OK EXIT",
                    warn_color,
                    0.92f,
                    2);
}

class VisionControlTcpServer {
public:
    explicit VisionControlTcpServer(int port)
        : port_(port)
    {
    }

    ~VisionControlTcpServer()
    {
        close_all();
    }

    bool start()
    {
        listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listen_fd_ < 0) {
            log::error("tcp server socket create failed: errno=%d(%s)", errno, std::strerror(errno));
            return false;
        }

        int reuse = 1;
        ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
        if (!set_non_blocking(listen_fd_)) {
            log::error("tcp server set nonblocking failed");
            close_all();
            return false;
        }

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(static_cast<uint16_t>(port_));
        if (::bind(listen_fd_, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) {
            log::error("tcp server bind failed: port=%d errno=%d(%s)", port_, errno, std::strerror(errno));
            close_all();
            return false;
        }

        if (::listen(listen_fd_, 1) != 0) {
            log::error("tcp server listen failed: errno=%d(%s)", errno, std::strerror(errno));
            close_all();
            return false;
        }

        log::info("vision tcp server listen at 0.0.0.0:%d", port_);
        return true;
    }

    void poll(const VisionStatusSnapshot &snapshot, PendingControl &control)
    {
        accept_client_if_needed();
        if (client_fd_ < 0) {
            return;
        }

        uint8_t buffer[4096];
        while (true) {
            const ssize_t count = ::recv(client_fd_, buffer, sizeof(buffer), 0);
            if (count > 0) {
                rx_buffer_.insert(rx_buffer_.end(), buffer, buffer + count);
                continue;
            }

            if (count == 0) {
                log::info("vision tcp client disconnected");
                close_client();
                return;
            }

            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                break;
            }

            log::warn("vision tcp recv failed: errno=%d(%s)", errno, std::strerror(errno));
            close_client();
            return;
        }

        while (true) {
            auto frame = decode_one_frame();
            if (!frame.has_value()) {
                break;
            }
            const auto response = process_request(frame.value(), snapshot, control);
            if (!response.empty()) {
                send_bytes(response);
            }
        }
    }

    bool is_listening() const
    {
        return listen_fd_ >= 0;
    }

    bool is_client_connected() const
    {
        return client_fd_ >= 0;
    }

    void push_status_periodic(const VisionStatusSnapshot &snapshot, uint64_t now_ms, uint64_t interval_ms)
    {
        if (client_fd_ < 0) {
            return;
        }
        if (interval_ms == 0) {
            return;
        }
        if ((now_ms - last_status_push_ms_) < interval_ms) {
            return;
        }

        auto body = encode_status_body(snapshot);
        auto frame = encode_resp_ok(APP_CMD_VISION_STATUS, body);
        send_bytes(frame);
        last_status_push_ms_ = now_ms;
    }

private:
    bool set_non_blocking(int fd)
    {
        const int flags = ::fcntl(fd, F_GETFL, 0);
        if (flags < 0) {
            return false;
        }
        return ::fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0;
    }

    void accept_client_if_needed()
    {
        if (listen_fd_ < 0 || client_fd_ >= 0) {
            return;
        }

        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        const int fd = ::accept(listen_fd_, reinterpret_cast<sockaddr *>(&client_addr), &client_len);
        if (fd < 0) {
            if (errno != EAGAIN && errno != EWOULDBLOCK) {
                log::warn("vision tcp accept failed: errno=%d(%s)", errno, std::strerror(errno));
            }
            return;
        }

        if (!set_non_blocking(fd)) {
            log::warn("vision tcp client set nonblocking failed");
            ::close(fd);
            return;
        }

        client_fd_ = fd;
        rx_buffer_.clear();

        char addr_text[64] = {0};
        ::inet_ntop(AF_INET, &client_addr.sin_addr, addr_text, sizeof(addr_text));
        log::info("vision tcp client connected: %s:%d", addr_text, ntohs(client_addr.sin_port));
    }

    std::optional<ProtocolFrame> decode_one_frame()
    {
        if (rx_buffer_.size() < 8) {
            return std::nullopt;
        }

        const uint8_t header_bytes[] = {
            static_cast<uint8_t>(kHeader & 0xFF),
            static_cast<uint8_t>((kHeader >> 8) & 0xFF),
            static_cast<uint8_t>((kHeader >> 16) & 0xFF),
            static_cast<uint8_t>((kHeader >> 24) & 0xFF)};

        if (!std::equal(header_bytes, header_bytes + 4, rx_buffer_.begin())) {
            auto it = std::search(rx_buffer_.begin(), rx_buffer_.end(), header_bytes, header_bytes + 4);
            if (it == rx_buffer_.end()) {
                rx_buffer_.clear();
                return std::nullopt;
            }
            rx_buffer_.erase(rx_buffer_.begin(), it);
            if (rx_buffer_.size() < 8) {
                return std::nullopt;
            }
        }

        const uint32_t data_len = read_u32_le(rx_buffer_.data() + 4);
        if (data_len < 4) {
            rx_buffer_.erase(rx_buffer_.begin(), rx_buffer_.begin() + 4);
            return std::nullopt;
        }

        const size_t full_len = 8 + data_len;
        if (rx_buffer_.size() < full_len) {
            return std::nullopt;
        }

        const uint16_t recv_crc = read_u16_le(rx_buffer_.data() + full_len - 2);
        const uint16_t calc_crc = crc16_ibm(rx_buffer_.data(), full_len - 2);
        if (recv_crc != calc_crc) {
            log::warn("vision tcp crc mismatch recv=0x%04X calc=0x%04X", recv_crc, calc_crc);
            rx_buffer_.erase(rx_buffer_.begin(), rx_buffer_.begin() + static_cast<long>(full_len));
            return std::nullopt;
        }

        ProtocolFrame frame;
        frame.flags = rx_buffer_[8];
        frame.cmd = rx_buffer_[9];
        frame.body.assign(rx_buffer_.begin() + 10, rx_buffer_.begin() + static_cast<long>(full_len - 2));
        rx_buffer_.erase(rx_buffer_.begin(), rx_buffer_.begin() + static_cast<long>(full_len));
        return frame;
    }

    std::vector<uint8_t> encode_frame(uint8_t flags, uint8_t cmd, const std::vector<uint8_t> &body)
    {
        std::vector<uint8_t> out;
        out.reserve(8 + 1 + 1 + body.size() + 2);
        append_u32_le(out, kHeader);
        append_u32_le(out, static_cast<uint32_t>(1 + 1 + body.size() + 2));
        out.push_back(flags);
        out.push_back(cmd);
        out.insert(out.end(), body.begin(), body.end());
        const uint16_t crc = crc16_ibm(out.data(), out.size());
        out.push_back(static_cast<uint8_t>(crc & 0xFF));
        out.push_back(static_cast<uint8_t>((crc >> 8) & 0xFF));
        return out;
    }

    std::vector<uint8_t> encode_resp_ok(uint8_t cmd, const std::vector<uint8_t> &body = {})
    {
        return encode_frame(kFlagIsResp | kFlagRespOk | kProtocolVersion, cmd, body);
    }

    std::vector<uint8_t> encode_resp_err(uint8_t cmd, uint8_t err_code, const std::string &err_text)
    {
        std::vector<uint8_t> body;
        body.push_back(err_code);
        body.insert(body.end(), err_text.begin(), err_text.end());
        return encode_frame(kFlagIsResp | kProtocolVersion, cmd, body);
    }

    std::vector<uint8_t> encode_app_info_body() const
    {
        std::vector<uint8_t> body;
        body.push_back(0);
        append_cstr(body, kAppId);
        append_cstr(body, kAppName);
        append_cstr(body, kAppDesc);
        return body;
    }

    std::vector<uint8_t> encode_status_body(const VisionStatusSnapshot &snapshot) const
    {
        std::vector<uint8_t> body;
        append_cstr(body, vision_app_state_to_text(snapshot.app_state));
        append_cstr(body, snapshot.tracking_state);
        append_cstr(body, snapshot.active ? "1" : "0");
        append_cstr(body, snapshot.target_found ? "1" : "0");
        append_cstr(body, snapshot.laser_found ? "1" : "0");
        append_cstr(body, snapshot.command);
        append_cstr(body, snapshot.tracking_enabled ? "1" : "0");
        append_cstr(body, snapshot.can_scan ? "1" : "0");
        append_cstr(body, std::to_string(snapshot.target_x));
        append_cstr(body, std::to_string(snapshot.target_y));
        return body;
    }

    bool matches_current_app(const std::vector<uint8_t> &body) const
    {
        if (body.empty()) {
            return true;
        }

        if (body[0] != 0xFF) {
            return true;
        }

        const auto fields = split_strings_with_nul(body, 1);
        if (fields.empty()) {
            return true;
        }

        for (const auto &field : fields) {
            if (field == kAppId || field == "id:" + std::string(kAppId)) {
                return true;
            }
        }
        return false;
    }

    std::vector<uint8_t> process_request(
        const ProtocolFrame &frame,
        const VisionStatusSnapshot &snapshot,
        PendingControl &control)
    {
        if ((frame.flags & kFlagIsResp) != 0U) {
            return {};
        }

        switch (frame.cmd) {
            case CMD_SET_REPORT:
                return encode_resp_ok(frame.cmd, {});
            case CMD_APP_LIST: {
                std::vector<uint8_t> body;
                body.push_back(1);
                append_cstr(body, kAppId);
                return encode_resp_ok(frame.cmd, body);
            }
            case CMD_CUR_APP_INFO:
            case CMD_APP_INFO:
                return encode_resp_ok(frame.cmd, encode_app_info_body());
            case CMD_START_APP:
                if (!matches_current_app(frame.body)) {
                    return encode_resp_err(frame.cmd, 1, "app_id not match");
                }
                control.recognition_start_requested = true;
                control.stop_requested = false;
                return encode_resp_ok(frame.cmd, {});
            case APP_CMD_VISION_START:
                if (!matches_current_app(frame.body)) {
                    return encode_resp_err(frame.cmd, 1, "app_id not match");
                }
                control.recognition_start_requested = true;
                control.tracking_start_requested = true;
                control.stop_requested = false;
                if (frame.body.size() >= 5 && frame.body[0] == 0xFE) {
                    const int yaw_pwm = static_cast<int>(read_u16_le(frame.body.data() + 1));
                    const int pitch_pwm = static_cast<int>(read_u16_le(frame.body.data() + 3));
                    control.init_yaw_pwm = std::clamp(yaw_pwm, 500, 2500);
                    control.init_pitch_pwm = std::clamp(pitch_pwm, 500, 2500);
                    control.tracking_init_pose_valid = true;
                }
                if (frame.body.size() >= 6 && frame.body[0] == 0xFE) {
                    control.tracking_start_requested = (frame.body[5] != 0U);
                }
                return encode_resp_ok(frame.cmd, {});
            case CMD_EXIT_APP:
            case APP_CMD_VISION_STOP:
                if (!matches_current_app(frame.body)) {
                    return encode_resp_err(frame.cmd, 1, "app_id not match");
                }
                control.stop_requested = true;
                control.recognition_start_requested = false;
                control.tracking_start_requested = false;
                return encode_resp_ok(frame.cmd, {});
            case APP_CMD_VISION_STATUS:
                return encode_resp_ok(frame.cmd, encode_status_body(snapshot));
            default:
                return encode_resp_err(frame.cmd, 2, "cmd not support");
        }
    }

    void send_bytes(const std::vector<uint8_t> &data)
    {
        if (client_fd_ < 0 || data.empty()) {
            return;
        }

        size_t offset = 0;
        const uint64_t start_ms = time::ticks_ms();
        while (offset < data.size()) {
            const ssize_t written = ::send(client_fd_, data.data() + offset, data.size() - offset, 0);
            if (written > 0) {
                offset += static_cast<size_t>(written);
                continue;
            }

            if (written < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
                // Avoid stalling vision loop on a slow peer; drop connection if socket stays blocked.
                if ((time::ticks_ms() - start_ms) > 3) {
                    log::warn("vision tcp send timeout, close slow client");
                    close_client();
                    return;
                }
                continue;
            }

            log::warn("vision tcp send failed: errno=%d(%s)", errno, std::strerror(errno));
            close_client();
            return;
        }
    }

    void close_client()
    {
        if (client_fd_ >= 0) {
            ::close(client_fd_);
            client_fd_ = -1;
        }
        rx_buffer_.clear();
    }

    void close_all()
    {
        close_client();
        if (listen_fd_ >= 0) {
            ::close(listen_fd_);
            listen_fd_ = -1;
        }
    }

    int port_{5555};
    int listen_fd_{-1};
    int client_fd_{-1};
    std::vector<uint8_t> rx_buffer_;
    uint64_t last_status_push_ms_{0};
};

} // namespace

int _main(int argc, char *argv[])
{
    int frame_width = 512;
    int frame_height = 512;
    int tcp_port = 5555;
    std::string cmd_log_path = "servo_command_debug.csv";
    bool invert_pitch = false;
    bool invert_yaw = false;
    float laser_offset_up_mm = 20.0f;
    float nominal_target_distance_mm = 800.0f;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--width" && i + 1 < argc) {
            frame_width = std::stoi(argv[++i]);
        } else if (arg == "--height" && i + 1 < argc) {
            frame_height = std::stoi(argv[++i]);
        } else if (arg == "--tcp-port" && i + 1 < argc) {
            tcp_port = std::stoi(argv[++i]);
        } else if (arg == "--cmd-log" && i + 1 < argc) {
            cmd_log_path = argv[++i];
        } else if (arg == "--invert-pitch") {
            invert_pitch = true;
        } else if (arg == "--invert-yaw") {
            invert_yaw = true;
        } else if (arg == "--laser-offset-up-mm" && i + 1 < argc) {
            laser_offset_up_mm = std::stof(argv[++i]);
        } else if (arg == "--target-distance-mm" && i + 1 < argc) {
            nominal_target_distance_mm = std::stof(argv[++i]);
        }
    }

    // Keep enough pixels for robust target detection while reducing processing load.
    frame_width = std::max(frame_width, 320);
    frame_height = std::max(frame_height, 320);

    std::string device_name = sys::device_name();
    std::transform(device_name.begin(), device_name.end(), device_name.begin(), ::tolower);
    log::info("device: %s", device_name.c_str());
    log::info("camera resolution: %dx%d", frame_width, frame_height);
    log::info("local uart forwarding disabled, use tcp->ros control path");

    const bool auto_start_tracking = true;

    TargetTrackingPipeline pipeline;
    PipelineConfig cfg = pipeline.get_config();
    cfg.fx = 381.625f;
    cfg.fy = 381.625f;
    cfg.cx = static_cast<float>(frame_width) * 0.5f;
    // Approximate mm->pixel conversion on image Y axis: px_per_mm = fy / Z(mm).
    // Move optical center upward to make aiming center align with laser hit point.
    const float px_per_mm_y = cfg.fy / std::max(1.0f, nominal_target_distance_mm);
    const float laser_offset_up_px = laser_offset_up_mm * px_per_mm_y;
    cfg.cy = static_cast<float>(frame_height) * 0.5f - laser_offset_up_px;
    cfg.cy = clamp_value(cfg.cy, 0.0f, static_cast<float>(frame_height - 1));
    // 与 test_gimbal_control 对齐：PWM=1500 对应 135deg（正前方）
    cfg.pitch_home = 135.0f;
    cfg.yaw_home = 135.0f;
    cfg.pitch_pwm_zero_angle = 135.0f;
    cfg.yaw_pwm_zero_angle = 135.0f;
    cfg.pid_kp = 3.6f;
    cfg.pid_ki = 0.0f;
    cfg.pid_kd = 0.18f;
    cfg.scan_yaw_freq = 0.08f;
    cfg.scan_pitch_freq = 0.10f;
    // Real device observation: larger yaw PWM turns camera left, larger pitch PWM turns camera up.
    // These signs map image error to that physical motion model.
    cfg.pitch_error_sign = 1.0f;
    cfg.yaw_error_sign = 1.0f;
    if (invert_pitch) {
        cfg.pitch_error_sign *= -1.0f;
    }
    if (invert_yaw) {
        cfg.yaw_error_sign *= -1.0f;
    }
    cfg.enable_serial = false;
    cfg.draw_overlay = false;
    cfg.print_debug = false;
    cfg.enable_view_angle_feedforward = true;
    cfg.enable_open_loop_phase_orbit = true;
    cfg.open_loop_omega_rad_s = 1.806f; // pi/3 rad/s
    cfg.open_loop_phase_init_rad = 0.0f;
    cfg.open_loop_default_distance_mm = nominal_target_distance_mm;
    cfg.enable_speed_identification = true;
    pipeline.set_config(cfg);
    log::info("aim center compensation: laser_offset_up=%.2fmm, distance=%.2fmm, px_per_mm=%.4f, cy_shift=%.2fpx, cy=%.2f",
              laser_offset_up_mm,
              nominal_target_distance_mm,
              px_per_mm_y,
              laser_offset_up_px,
              cfg.cy);

    TrackerConfig tracker_cfg = pipeline.get_tracker_config();
    tracker_cfg.show_debug_windows = false;
    tracker_cfg.print_debug_info = false;
    tracker_cfg.camera_fx_px = cfg.fx;
    tracker_cfg.camera_fy_px = cfg.fy;
    tracker_cfg.enable_board_distance_estimation = true;
    tracker_cfg.board_distance_calibration_scale = 1.6f;
    pipeline.set_tracker_config(tracker_cfg);

    VisionControlTcpServer tcp_server(tcp_port);
    if (!tcp_server.start()) {
        return -1;
    }

    camera::Camera cam(frame_width, frame_height, image::Format::FMT_RGB888);
    display::Display disp;
    peripheral::key::add_default_listener();
    log::info("local OK key exit enabled");

    std::ofstream cmd_log(cmd_log_path, std::ios::out | std::ios::trunc);
    if (!cmd_log.is_open()) {
        log::warn("failed to open command log file: %s", cmd_log_path.c_str());
    } else {
        cmd_log << "time_ms,frame,app_state,track_state,active,target_found,laser_found,command\n";
        log::info("command log file: %s", cmd_log_path.c_str());
    }

    VisionAppState app_state = VisionAppState::Idle;
    bool recognition_active = false;
    bool tracking_enabled = false;
    uint64_t run_start_ms = time::ticks_ms();
    uint64_t frame_index = 0;
    uint64_t last_tick_ms = time::ticks_ms();
    TrackState last_state = TrackState::Waiting;
    PipelineOutput last_output;
    last_output.state = TrackState::Waiting;
    last_output.command.clear();

    constexpr int kStreamPort = 8000;
    constexpr uint64_t kStreamIntervalMs = 300;
    constexpr uint64_t kCmdLogFlushIntervalMs = 500;
    constexpr uint64_t kInfoLogIntervalMs = 500;
    constexpr uint64_t kStatusPushIntervalMs = 30;
    http::JpegStreamer stream("", kStreamPort);
    stream.set_html(kStreamHtml);
    bool stream_active = false;
    log::info("background stream standby: http://%s:%d/stream", stream.host().c_str(), stream.port());
    uint64_t last_stream_ms = time::ticks_ms();
    uint64_t last_cmd_log_flush_ms = last_stream_ms;
    uint64_t last_distance_log_ms = 0;
    uint64_t last_speed_validate_log_ms = 0;
    uint64_t last_open_loop_log_ms = 0;

    while (!app::need_exit()) {
        PendingControl control;
        VisionStatusSnapshot snapshot;
        snapshot.app_state = app_state;
        snapshot.tracking_state = recognition_active ? state_to_text(last_output.state) : "Stopped";
        snapshot.active = recognition_active;
        snapshot.tracking_enabled = tracking_enabled;
        snapshot.can_scan = recognition_active && !tracking_enabled;
        snapshot.target_found = recognition_active ? last_output.target_found : false;
        snapshot.laser_found = recognition_active ? last_output.laser_found : false;
        snapshot.target_x = (recognition_active && last_output.target_found)
                    ? static_cast<int>(std::lround(last_output.target_pos.x))
                    : -1;
        snapshot.target_y = (recognition_active && last_output.target_found)
                    ? static_cast<int>(std::lround(last_output.target_pos.y))
                    : -1;
        snapshot.command = tracking_enabled ? last_output.command : "";
        tcp_server.poll(snapshot, control);

        if (control.stop_requested) {
            recognition_active = false;
            tracking_enabled = false;
            app_state = VisionAppState::Stopped;
            pipeline.reset();
            pipeline.set_control_enabled(false);
            last_output = PipelineOutput();
            last_output.state = TrackState::Waiting;
            last_output.command.clear();
            last_state = TrackState::Waiting;
            if (stream_active) {
                stream.stop();
                stream_active = false;
                log::info("background stream stopped (vision stop)");
            }
            log::info("vision task stopped by tcp command");
        }

        if (control.tracking_start_requested) {
            if (!recognition_active) {
                pipeline.reset();
                pipeline.set_control_enabled(false);
                pipeline.handle_key(' ');
                recognition_active = true;
                last_state = TrackState::Waiting;
            }
            if (!tracking_enabled) {
                tracking_enabled = true;
                pipeline.set_control_enabled(true);
                if (control.tracking_init_pose_valid) {
                    const float yaw_deg = pwm_to_angle_deg(control.init_yaw_pwm, cfg.yaw_pwm_zero_angle);
                    const float pitch_deg = pwm_to_angle_deg(control.init_pitch_pwm, cfg.pitch_pwm_zero_angle);
                    pipeline.set_current_angles(pitch_deg, yaw_deg);
                    log::info("tracking init pose from host pwm: yaw=%d(%.2fdeg) pitch=%d(%.2fdeg)",
                              control.init_yaw_pwm,
                              yaw_deg,
                              control.init_pitch_pwm,
                              pitch_deg);
                }
                pipeline.start_tracking();
                app_state = VisionAppState::Running;
                if (stream_active) {
                    stream.stop();
                    stream_active = false;
                    log::info("background stream stopped (tracking start)");
                }
                log::info("vision tracking enabled by tcp command (open_loop=%d, omega=%.4f rad/s)",
                          cfg.enable_open_loop_phase_orbit ? 1 : 0,
                          cfg.open_loop_omega_rad_s);
            } else {
                pipeline.set_control_enabled(true);
                if (control.tracking_init_pose_valid) {
                    log::info("ignore duplicate tracking-start init pose while tracking is active");
                }
            }
        } else if (control.recognition_start_requested) {
            pipeline.reset();
            pipeline.set_control_enabled(true);
            pipeline.handle_key(' ');
            recognition_active = true;
            tracking_enabled = false;
            app_state = VisionAppState::Running;
            last_state = TrackState::Waiting;
            if (!stream_active) {
                stream.start();
                stream_active = true;
                log::info("background stream started (recognition mode)");
            }
            log::info("vision task started by tcp command");
        }

        image::Image *img = cam.read();
        if (!img) {
            time::sleep_ms(5);
            continue;
        }

        uint64_t now_tick_ms = time::ticks_ms();
        frame_index++;
        float dt = static_cast<float>(now_tick_ms - last_tick_ms) / 1000.0f;
        last_tick_ms = now_tick_ms;
        dt = clamp_value(dt, 0.001f, 0.05f);

        PipelineOutput out;
        out.state = TrackState::Waiting;
        out.command.clear();

        if (recognition_active) {
            cv::Mat frame;
            maix::image::image2cv(*img, frame, true, true);
            pipeline.set_control_enabled(recognition_active);
            out = pipeline.process_frame(frame, dt);

            if (!tracking_enabled && out.target_found && out.board_distance_mm > 0.0f) {
                if ((now_tick_ms - last_distance_log_ms) >= kInfoLogIntervalMs) {
                    log::info("global recognition distance estimate: %.1f mm (target=(%.1f, %.1f), board=(%.1f, %.1f))",
                              out.board_distance_mm,
                              out.target_pos.x,
                              out.target_pos.y,
                              out.board_pos.x,
                              out.board_pos.y);
                    if (out.view_angle_valid) {
                        log::info("feedforward servo angles: pitch=%.2f deg yaw=%.2f deg (d_pitch=%.4f rad, d_yaw=%.4f rad)",
                                  out.feedforward_pitch_angle,
                                  out.feedforward_yaw_angle,
                                  out.view_delta_pitch_rad,
                                  out.view_delta_yaw_rad);
                    }
                    last_distance_log_ms = now_tick_ms;
                }
            }

            if (!tracking_enabled && out.speed_identifying && out.predicted_pos_valid) {
                if ((now_tick_ms - last_speed_validate_log_ms) >= kInfoLogIntervalMs) {
                    log::info("speed-id validating: inst=%.4f fit=%.4f rad/s n=%d err=%.2f/%.2f px",
                              out.instant_omega_rad_s,
                              out.fitted_omega_rad_s,
                              out.speed_fit_samples,
                              out.speed_validation_error_px,
                              out.speed_validation_tolerance_px);
                    last_speed_validate_log_ms = now_tick_ms;
                }
            }

            if (!tracking_enabled && out.speed_identified_event) {
                log::info("speed-id success -> LOCKED (omega=%.4f rad/s), waiting tracking start",
                          out.identified_omega_rad_s);
            }

            if (out.state != last_state) {
                log::info("[STATE] %s -> %s",
                          state_to_text(last_state),
                          state_to_text(out.state));
            }

            if (auto_start_tracking && tracking_enabled && out.state == TrackState::Locked) {
                pipeline.start_tracking();
                log::info("[AUTO] Locked -> Tracking");
            }

            if (tracking_enabled && out.open_loop_active) {
                if ((now_tick_ms - last_open_loop_log_ms) >= kInfoLogIntervalMs) {
                    log::info("open-loop orbit: phase=%.3f rad omega=%.3f rad/s dist=%.1f mm pitch=%.2f yaw=%.2f",
                              out.open_loop_phase_rad,
                              out.open_loop_omega_rad_s,
                              out.open_loop_distance_mm,
                              out.pitch_angle,
                              out.yaw_angle);
                    last_open_loop_log_ms = now_tick_ms;
                }
            }
            last_state = out.state;
        } else {
            out.state = TrackState::Waiting;
            out.command.clear();
        }

        last_output = out;

        VisionStatusSnapshot live_snapshot;
        live_snapshot.app_state = app_state;
        live_snapshot.tracking_state = recognition_active ? state_to_text(out.state) : "Stopped";
        live_snapshot.active = recognition_active;
        live_snapshot.tracking_enabled = tracking_enabled;
        live_snapshot.can_scan = recognition_active && !tracking_enabled;
        live_snapshot.target_found = recognition_active ? out.target_found : false;
        live_snapshot.laser_found = recognition_active ? out.laser_found : false;
        live_snapshot.target_x = (recognition_active && out.target_found)
                    ? static_cast<int>(std::lround(out.target_pos.x))
                    : -1;
        live_snapshot.target_y = (recognition_active && out.target_found)
                    ? static_cast<int>(std::lround(out.target_pos.y))
                    : -1;
        live_snapshot.command = tracking_enabled ? out.command : "";
        tcp_server.push_status_periodic(live_snapshot, now_tick_ms, kStatusPushIntervalMs);

        if (out.tracking_recovery_requested) {
            tracking_enabled = false;
            pipeline.set_control_enabled(true);
            log::info("tracking lost -> reset to recognition/home, wait for next A");
        }

        if (cmd_log.is_open()) {
            std::string command_clean = out.command;
            for (char &ch : command_clean) {
                if (ch == '"') {
                    ch = '\'';
                }
            }
            cmd_log << (now_tick_ms - run_start_ms) << ","
                    << frame_index << ","
                    << vision_app_state_to_text(app_state) << ","
                    << state_to_text(out.state) << ","
                    << (recognition_active ? 1 : 0) << ","
                    << (out.target_found ? 1 : 0) << ","
                    << (out.laser_found ? 1 : 0) << ",\""
                    << command_clean << "\"\n";
            if ((now_tick_ms - last_cmd_log_flush_ms) >= kCmdLogFlushIntervalMs) {
                cmd_log.flush();
                last_cmd_log_flush_ms = now_tick_ms;
            }
        }

        draw_pipeline_overlay(*img,
                              out,
                              app_state,
                              recognition_active,
                              tcp_server.is_listening(),
                              tcp_server.is_client_connected(),
                              tcp_port);

        if (stream_active && recognition_active && tcp_server.is_client_connected() &&
            (now_tick_ms - last_stream_ms) >= kStreamIntervalMs) {
            image::Image *jpg = img->to_jpeg();
            if (jpg) {
                stream.write(jpg);
                delete jpg;
            }
            last_stream_ms = now_tick_ms;
        }

        disp.show(*img, image::FIT_COVER);
        delete img;
    }

    if (stream_active) {
        stream.stop();
    }

    return 0;
}

int main(int argc, char *argv[])
{
    sys::register_default_signal_handle();
    CATCH_EXCEPTION_RUN_RETURN(_main, -1, argc, argv);
}
