#include "maix_basic.hpp"
#include "maix_camera.hpp"
#include "maix_display.hpp"
#include "maix_image_cv.hpp"
#include "maix_pinmap.hpp"
#include "main.h"

#include "TargetTracking/TargetTrackingPipeline.hpp"

#include <algorithm>
#include <arpa/inet.h>
#include <cerrno>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
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
    bool target_found{false};
    bool laser_found{false};
    std::string command;
};

struct PendingControl {
    bool start_requested{false};
    bool stop_requested{false};
};

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

bool is_bluetooth_serial_device(const std::string &device_name)
{
    return device_name.rfind("/dev/rfcomm", 0) == 0;
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

void setup_uart_pinmux(const std::string &device_name, const std::string &uart_device)
{
    auto set_pin = [](const std::string &pin_name, const std::string &function_name) {
        err::Err ret = peripheral::pinmap::set_pin_function(pin_name.c_str(), function_name.c_str());
        if (ret != err::Err::ERR_NONE) {
            log::warn("pinmux failed: %s -> %s", pin_name.c_str(), function_name.c_str());
            return false;
        }
        return true;
    };

    if (device_name == "maixcam2") {
        if (uart_device != "/dev/ttyS4") {
            log::warn("MaixCAM2 recommends UART4 on A21/A22, current device: %s", uart_device.c_str());
        }
        set_pin("A21", "UART4_TX");
        set_pin("A22", "UART4_RX");
        return;
    }

    if (uart_device == "/dev/ttyS0") {
        set_pin("A16", "UART0_TX");
        set_pin("A17", "UART0_RX");
    }
}

void draw_pipeline_overlay(
    image::Image &img,
    const PipelineOutput &out,
    VisionAppState app_state,
    bool task_active)
{
    const image::Color target_color = image::Color::from_rgb(255, 64, 64);
    const image::Color laser_color = image::Color::from_rgb(255, 255, 0);
    const image::Color info_color = image::Color::from_rgb(0, 255, 0);
    const image::Color ok_color = image::Color::from_rgb(0, 255, 0);
    const image::Color bad_color = image::Color::from_rgb(255, 64, 64);
    const image::Color hit_color = image::Color::from_rgb(0, 255, 0);
    const image::Color miss_color = image::Color::from_rgb(255, 180, 0);
    constexpr float kHitThresholdPx = 30.0f;

    char line0[128] = {0};
    snprintf(
        line0,
        sizeof(line0),
        "task:%s app:%s",
        task_active ? "ACTIVE" : "WAIT_START",
        vision_app_state_to_text(app_state));
    img.draw_string(6, 6, line0, info_color, 1.2f);

    std::string line1 = std::string("state:") + state_to_text(out.state) +
                        " lock:" + std::to_string(out.lock_count) +
                        " lost:" + std::to_string(out.lost_count);
    img.draw_string(6, 28, line1, info_color, 1.2f);

    char line2[128] = {0};
    snprintf(line2,
             sizeof(line2),
             "pitch:%.2f yaw:%.2f",
             out.pitch_angle,
             out.yaw_angle);
    img.draw_string(6, 50, line2, info_color, 1.2f);

    if (!task_active) {
        img.draw_string(6, 72, "TCP control ready, wait host START", image::Color::from_rgb(0, 200, 255), 1.2f);
        img.draw_string(6, 94, "Host STOP keeps app alive", image::Color::from_rgb(0, 200, 255), 1.2f);
        return;
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

    if (out.target_found && out.laser_found) {
        const bool detect_hit = out.laser_target_error_px < kHitThresholdPx;
        img.draw_line(static_cast<int>(out.target_pos.x),
                      static_cast<int>(out.target_pos.y),
                      static_cast<int>(out.laser_pos.x),
                      static_cast<int>(out.laser_pos.y),
                      detect_hit ? hit_color : miss_color,
                      2);
        char line3[128] = {0};
        snprintf(line3,
                 sizeof(line3),
                 "detect:%s err=%.2fpx",
                 detect_hit ? "HIT" : "MISS",
                 out.laser_target_error_px);
        img.draw_string(6, 72, line3, detect_hit ? hit_color : miss_color, 1.2f);
    } else {
        img.draw_string(6, 72, "detect:N/A", miss_color, 1.2f);
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

    std::string target_status = std::string("target:") + (out.target_found ? "FOUND" : "NOT FOUND");
    std::string laser_status = std::string("laser:") + (out.laser_found ? "FOUND" : "NOT FOUND");
    img.draw_string(6, 94, target_status, out.target_found ? ok_color : bad_color, 1.2f);
    img.draw_string(6, 116, laser_status, out.laser_found ? ok_color : bad_color, 1.2f);
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
            case APP_CMD_VISION_START:
                if (!matches_current_app(frame.body)) {
                    return encode_resp_err(frame.cmd, 1, "app_id not match");
                }
                control.start_requested = true;
                control.stop_requested = false;
                return encode_resp_ok(frame.cmd, {});
            case CMD_EXIT_APP:
            case APP_CMD_VISION_STOP:
                if (!matches_current_app(frame.body)) {
                    return encode_resp_err(frame.cmd, 1, "app_id not match");
                }
                control.stop_requested = true;
                control.start_requested = false;
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
        while (offset < data.size()) {
            const ssize_t written = ::send(client_fd_, data.data() + offset, data.size() - offset, 0);
            if (written > 0) {
                offset += static_cast<size_t>(written);
                continue;
            }

            if (written < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
                time::sleep_ms(1);
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
};

} // namespace

int _main(int argc, char *argv[])
{
    int frame_width = 640;
    int frame_height = 480;
    bool enable_pipeline_uart = false;
    int uart_baud = 115200;
    int tcp_port = 5555;
    std::string uart_device = "/dev/ttyS4";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--width" && i + 1 < argc) {
            frame_width = std::stoi(argv[++i]);
        } else if (arg == "--height" && i + 1 < argc) {
            frame_height = std::stoi(argv[++i]);
        } else if (arg == "--uart" && i + 1 < argc) {
            uart_device = argv[++i];
            enable_pipeline_uart = true;
        } else if (arg == "--bt-rfcomm" && i + 1 < argc) {
            uart_device = argv[++i];
            enable_pipeline_uart = true;
        } else if (arg == "--baud" && i + 1 < argc) {
            uart_baud = std::stoi(argv[++i]);
        } else if (arg == "--tcp-port" && i + 1 < argc) {
            tcp_port = std::stoi(argv[++i]);
        } else if (arg == "--no-uart") {
            enable_pipeline_uart = false;
        }
    }

    std::string device_name = sys::device_name();
    std::transform(device_name.begin(), device_name.end(), device_name.begin(), ::tolower);
    log::info("device: %s", device_name.c_str());

    if (device_name != "maixcam2" && uart_device == "/dev/ttyS4") {
        uart_device = "/dev/ttyS0";
        log::info("non-MaixCAM2 platform, fallback uart device to %s", uart_device.c_str());
    }

    if (enable_pipeline_uart && !is_bluetooth_serial_device(uart_device)) {
        setup_uart_pinmux(device_name, uart_device);
    } else if (enable_pipeline_uart && is_bluetooth_serial_device(uart_device)) {
        log::info("bluetooth serial output enabled: %s", uart_device.c_str());
    } else {
        log::info("local uart forwarding disabled, use tcp control path");
    }

    const bool auto_start_tracking = true;

    TargetTrackingPipeline pipeline;
    PipelineConfig cfg = pipeline.get_config();
    cfg.fx = -1.0f;
    cfg.fy = -1.0f;
    cfg.cx = -1.0f;
    cfg.cy = -1.0f;
    cfg.pitch_home = 60.0f;
    cfg.yaw_home = 105.0f;
    cfg.pitch_pwm_zero_angle = cfg.pitch_home;
    cfg.yaw_pwm_zero_angle = cfg.yaw_home;
    cfg.pitch_error_sign = -1.0f;
    cfg.yaw_error_sign = 1.0f;
    cfg.max_speed = 180.0f;
    cfg.integral_limit = 30.0f;
    cfg.enable_serial = enable_pipeline_uart;
    cfg.serial_device = uart_device;
    cfg.serial_baud = uart_baud;
    cfg.draw_overlay = false;
    cfg.print_debug = false;
    pipeline.set_config(cfg);

    TrackerConfig tracker_cfg = pipeline.get_tracker_config();
    tracker_cfg.show_debug_windows = false;
    tracker_cfg.print_debug_info = false;
    pipeline.set_tracker_config(tracker_cfg);

    if (enable_pipeline_uart) {
        bool opened = pipeline.open_serial();
        log::info("pipeline serial open: %s, dev=%s, baud=%d",
                  opened ? "true" : "false",
                  uart_device.c_str(),
                  uart_baud);
    }

    VisionControlTcpServer tcp_server(tcp_port);
    if (!tcp_server.start()) {
        return -1;
    }

    camera::Camera cam(frame_width, frame_height, image::Format::FMT_RGB888);
    display::Display disp;

    VisionAppState app_state = VisionAppState::Idle;
    bool task_active = false;
    uint64_t last_tick_ms = time::ticks_ms();
    TrackState last_state = TrackState::Waiting;
    PipelineOutput last_output;
    last_output.state = TrackState::Waiting;
    last_output.command.clear();

    while (!app::need_exit()) {
        PendingControl control;
        VisionStatusSnapshot snapshot;
        snapshot.app_state = app_state;
        snapshot.tracking_state = task_active ? state_to_text(last_output.state) : "Stopped";
        snapshot.active = task_active;
        snapshot.target_found = task_active ? last_output.target_found : false;
        snapshot.laser_found = task_active ? last_output.laser_found : false;
        snapshot.command = task_active ? last_output.command : "";
        tcp_server.poll(snapshot, control);

        if (control.stop_requested) {
            task_active = false;
            app_state = VisionAppState::Stopped;
            pipeline.reset();
            last_output = PipelineOutput();
            last_output.state = TrackState::Waiting;
            last_output.command.clear();
            last_state = TrackState::Waiting;
            log::info("vision task stopped by tcp command");
        }

        if (control.start_requested) {
            pipeline.reset();
            pipeline.handle_key(' ');
            task_active = true;
            app_state = VisionAppState::Running;
            last_state = TrackState::Waiting;
            log::info("vision task started by tcp command");
        }

        image::Image *img = cam.read();
        if (!img) {
            time::sleep_ms(5);
            continue;
        }

        uint64_t now_tick_ms = time::ticks_ms();
        float dt = static_cast<float>(now_tick_ms - last_tick_ms) / 1000.0f;
        last_tick_ms = now_tick_ms;
        dt = clamp_value(dt, 0.001f, 0.05f);

        PipelineOutput out;
        out.state = TrackState::Waiting;
        out.command.clear();

        if (task_active) {
            cv::Mat frame;
            maix::image::image2cv(*img, frame, true, true);
            out = pipeline.process_frame(frame, dt);

            if (out.state != last_state) {
                log::info("[STATE] %s -> %s",
                          state_to_text(last_state),
                          state_to_text(out.state));
            }

            if (auto_start_tracking && out.state == TrackState::Locked) {
                pipeline.handle_key(' ');
                log::info("[AUTO] Locked -> Tracking");
            }
            last_state = out.state;
        } else {
            out.state = TrackState::Waiting;
            out.command.clear();
        }

        last_output = out;
        draw_pipeline_overlay(*img, out, app_state, task_active);
        disp.show(*img, image::FIT_COVER);
        delete img;
    }

    pipeline.close_serial();
    return 0;
}

int main(int argc, char *argv[])
{
    sys::register_default_signal_handle();
    CATCH_EXCEPTION_RUN_RETURN(_main, -1, argc, argv);
}
