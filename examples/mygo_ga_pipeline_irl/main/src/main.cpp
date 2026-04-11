#include "maix_basic.hpp"
#include "maix_camera.hpp"
#include "maix_display.hpp"
#include "maix_image_cv.hpp"
#include "maix_jpg_stream.hpp"
#include "maix_key.hpp"
#include "main.h"

#include "TargetTracking/GeneticAlgorithm.hpp"
#include "TargetTracking/TargetTrackingPipeline.hpp"

#include <algorithm>
#include <arpa/inet.h>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <netinet/in.h>
#include <optional>
#include <sstream>
#include <string>
#include <sys/socket.h>
#include <sys/stat.h>
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
constexpr const char *kAppName = "MyGo GA Pipeline IRL";
constexpr const char *kAppDesc = "Online GA PID training with TCP control/status";

constexpr int kPopulationSize = 10;
constexpr int kGenerationCount = 20;
constexpr float kSampleDurationSec = 30.0f;
constexpr float kPidPMin = 2.0f;
constexpr float kPidPMax = 3.5f;
constexpr float kCenterHitThresholdPx = 18.0f;
constexpr float kLaserHitThresholdPx = 30.0f;
constexpr int kLostReturnToStartFrames = 4;
constexpr float kMutationRateStart = 0.16f;
constexpr float kMutationRateEnd = 0.06f;
constexpr float kMutationSigmaStart = 0.06f;
constexpr float kMutationSigmaEnd = 0.02f;

enum class VisionAppState {
    Idle,
    Running,
    Stopped,
};

enum class TrainingPhase {
    Idle,
    PrepareSample,
    TrackingSample,
    Finished,
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
    bool training_active{false};
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

struct EpisodeMetrics {
    int total_frames{0};
    int target_found_frames{0};
    int center_hit_frames{0};
    int laser_hit_frames{0};
    float sum_center_dist_px{0.0f};
    float sum_laser_target_dist_px{0.0f};
};

struct EvaluatedSample {
    Genome genome;
    EpisodeMetrics metrics;
    float fitness{-1e9f};
    int generation{0};
    int sample_index{0};
};

float clamp_value(float value, float low, float high)
{
    return std::max(low, std::min(value, high));
}

float pwm_to_angle_deg(int pwm, float zero_angle_deg)
{
    constexpr float kPwmPerDeg = 2000.0f / 270.0f;
    const float delta_deg = (static_cast<float>(pwm) - 1500.0f) / kPwmPerDeg;
    return std::clamp(zero_angle_deg + delta_deg, 0.0f, 270.0f);
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

        log::info("ga tcp server listen at 0.0.0.0:%d", port_);
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
                close_client();
                return;
            }

            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                break;
            }

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

    bool is_client_connected() const
    {
        return client_fd_ >= 0;
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
            return;
        }

        if (!set_non_blocking(fd)) {
            ::close(fd);
            return;
        }

        client_fd_ = fd;
        rx_buffer_.clear();
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
        append_cstr(body, snapshot.training_active ? "1" : "0");
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

    std::vector<uint8_t> process_request(const ProtocolFrame &frame,
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

bool ensure_dir(const std::string &path)
{
    struct stat st{};
    if (::stat(path.c_str(), &st) == 0) {
        return S_ISDIR(st.st_mode);
    }
    return ::mkdir(path.c_str(), 0755) == 0;
}

std::vector<Genome> evolve_population(const std::vector<EvaluatedSample> &evaluated,
                                      std::mt19937 &rng,
                                      int next_size,
                                      int elite_count,
                                      float mutation_rate,
                                      float mutation_sigma)
{
    std::vector<EvaluatedSample> sorted = evaluated;
    std::sort(sorted.begin(), sorted.end(), [](const EvaluatedSample &a, const EvaluatedSample &b) {
        return a.fitness > b.fitness;
    });

    std::vector<Genome> next;
    next.reserve(next_size);

    const int elite = std::min(elite_count, static_cast<int>(sorted.size()));
    for (int i = 0; i < elite; ++i) {
        next.push_back(sorted[i].genome);
    }

    auto tournament_pick = [&](int k) -> const Genome & {
        std::uniform_int_distribution<int> pick(0, static_cast<int>(sorted.size()) - 1);
        int best_idx = pick(rng);
        for (int i = 1; i < k; ++i) {
            int idx = pick(rng);
            if (sorted[idx].fitness > sorted[best_idx].fitness) {
                best_idx = idx;
            }
        }
        return sorted[best_idx].genome;
    };

    while (static_cast<int>(next.size()) < next_size) {
        const Genome &a = tournament_pick(3);
        const Genome &b = tournament_pick(3);
        Genome child = Genome::crossover(a, b, rng);
        child.mutate(rng, mutation_rate, mutation_sigma);
        next.push_back(child);
    }

    return next;
}

float compute_fitness(const EpisodeMetrics &m)
{
    if (m.total_frames <= 0) {
        return -1e9f;
    }

    const float total_frames_f = static_cast<float>(m.total_frames);
    const float target_found_ratio = static_cast<float>(m.target_found_frames) / total_frames_f;
    const float center_hit_ratio = static_cast<float>(m.center_hit_frames) / total_frames_f;
    const float laser_hit_ratio = static_cast<float>(m.laser_hit_frames) / total_frames_f;

    const float mean_center_dist = (m.target_found_frames > 0)
        ? (m.sum_center_dist_px / static_cast<float>(m.target_found_frames))
        : 9999.0f;
    const float mean_laser_dist = (m.laser_hit_frames > 0)
        ? (m.sum_laser_target_dist_px / static_cast<float>(m.laser_hit_frames))
        : 9999.0f;

    float penalty = 0.0f;
    if (target_found_ratio < 0.50f) {
        penalty += (0.50f - target_found_ratio) * 220.0f;
    }
    if (center_hit_ratio < 0.10f) {
        penalty += (0.10f - center_hit_ratio) * 180.0f;
    }
    if (m.target_found_frames < 50) {
        penalty += 80.0f;
    }

    const float score =
        260.0f * center_hit_ratio +
        140.0f * laser_hit_ratio +
        90.0f * target_found_ratio -
        1.6f * mean_center_dist -
        0.2f * std::min(mean_laser_dist, 1000.0f) -
        penalty;

    return score;
}

void draw_training_overlay(image::Image &img,
                           const PipelineOutput &out,
                           int generation_idx,
                           int sample_idx,
                           TrainingPhase phase,
                           const EpisodeMetrics &metrics,
                           const Genome &g)
{
    img.draw_string(8, 8,
                    "GA online training",
                    image::COLOR_WHITE,
                    1.4f);
    img.draw_string(8, 30,
                    "generation: " + std::to_string(generation_idx + 1) + "/" + std::to_string(kGenerationCount),
                    image::COLOR_GREEN,
                    1.2f);
    img.draw_string(8, 50,
                    "sample: " + std::to_string(sample_idx + 1) + "/" + std::to_string(kPopulationSize),
                    image::COLOR_GREEN,
                    1.2f);

    std::string phase_text = "idle";
    if (phase == TrainingPhase::PrepareSample) phase_text = "prepare";
    else if (phase == TrainingPhase::TrackingSample) phase_text = "tracking";
    else if (phase == TrainingPhase::Finished) phase_text = "finished";

    img.draw_string(8, 70, "phase: " + phase_text, image::COLOR_YELLOW, 1.2f);
    img.draw_string(8, 90,
                    "pid: p=" + std::to_string(g.p).substr(0, 6) +
                    " i=" + std::to_string(g.i).substr(0, 6) +
                    " d=" + std::to_string(g.d).substr(0, 6),
                    image::Color::from_rgb(0, 200, 255),
                    1.0f);

    const float dist_center = out.target_found
        ? std::sqrt((out.target_pos.x - out.aim_pos.x) * (out.target_pos.x - out.aim_pos.x) +
                    (out.target_pos.y - out.aim_pos.y) * (out.target_pos.y - out.aim_pos.y))
        : -1.0f;

    img.draw_string(8, 112,
                    "center_dist_px: " + (dist_center >= 0.0f ? std::to_string(dist_center).substr(0, 6) : "N/A"),
                    image::COLOR_WHITE,
                    1.0f);

    img.draw_string(8, 132,
                    "frames: total=" + std::to_string(metrics.total_frames) +
                    " found=" + std::to_string(metrics.target_found_frames) +
                    " center_hit=" + std::to_string(metrics.center_hit_frames),
                    image::COLOR_WHITE,
                    1.0f);

    if (out.target_found && out.target_pos.x >= 0.0f && out.target_pos.y >= 0.0f) {
        img.draw_rect(static_cast<int>(out.target_pos.x) - 3,
                      static_cast<int>(out.target_pos.y) - 3,
                      7,
                      7,
                      image::Color::from_rgb(255, 64, 64),
                      2);
        img.draw_string(static_cast<int>(out.target_pos.x) + 6,
                        static_cast<int>(out.target_pos.y) - 10,
                        "target",
                        image::Color::from_rgb(255, 64, 64),
                        1.0f);
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
                      image::Color::from_rgb(255, 255, 0),
                      2);
        img.draw_string(static_cast<int>(out.laser_pos.x) + 6,
                        static_cast<int>(out.laser_pos.y) - 10,
                        "laser",
                        image::Color::from_rgb(255, 255, 0),
                        1.0f);
    }

    if (out.target_found && out.laser_found) {
        img.draw_line(static_cast<int>(out.target_pos.x),
                      static_cast<int>(out.target_pos.y),
                      static_cast<int>(out.laser_pos.x),
                      static_cast<int>(out.laser_pos.y),
                      image::Color::from_rgb(0, 220, 120),
                      2);
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
}

void write_generation_report(const std::string &log_dir,
                             int generation_idx,
                             const std::vector<EvaluatedSample> &evaluated)
{
    if (evaluated.empty()) {
        return;
    }

    std::vector<EvaluatedSample> sorted = evaluated;
    std::sort(sorted.begin(), sorted.end(), [](const EvaluatedSample &a, const EvaluatedSample &b) {
        return a.fitness > b.fitness;
    });

    const EvaluatedSample &best = sorted.front();
    const float mean_center_dist = (best.metrics.target_found_frames > 0)
        ? best.metrics.sum_center_dist_px / static_cast<float>(best.metrics.target_found_frames)
        : 9999.0f;
    const float mean_laser_dist = (best.metrics.laser_hit_frames > 0)
        ? best.metrics.sum_laser_target_dist_px / static_cast<float>(best.metrics.laser_hit_frames)
        : 9999.0f;

    std::ostringstream filename;
    filename << log_dir << "/generation_" << std::setw(2) << std::setfill('0')
             << (generation_idx + 1) << "_report.txt";

    std::ofstream out(filename.str(), std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        return;
    }

    out << "generation=" << (generation_idx + 1) << "\n";
    out << "best_fitness=" << best.fitness << "\n";
    out << "best_pid_p=" << best.genome.p << "\n";
    out << "best_pid_i=" << best.genome.i << "\n";
    out << "best_pid_d=" << best.genome.d << "\n";
    out << "best_mean_center_distance_px=" << mean_center_dist << "\n";
    out << "best_mean_laser_target_distance_px=" << mean_laser_dist << "\n";
    out << "best_total_frames=" << best.metrics.total_frames << "\n";
    out << "best_target_found_frames=" << best.metrics.target_found_frames << "\n";
    out << "best_center_hit_frames=" << best.metrics.center_hit_frames << "\n";
    out << "best_laser_hit_frames=" << best.metrics.laser_hit_frames << "\n";

    out << "\nranked_samples:\n";
    for (size_t i = 0; i < sorted.size(); ++i) {
        const auto &s = sorted[i];
        const float d = (s.metrics.target_found_frames > 0)
            ? s.metrics.sum_center_dist_px / static_cast<float>(s.metrics.target_found_frames)
            : 9999.0f;
        out << "#" << (i + 1)
            << " fitness=" << s.fitness
            << " p=" << s.genome.p
            << " i=" << s.genome.i
            << " d=" << s.genome.d
            << " mean_center_dist=" << d
            << "\n";
    }
}

void append_best_pid(const std::string &log_dir, int generation_idx, const EvaluatedSample &best)
{
    std::ofstream out(log_dir + "/best_pid_each_generation.csv", std::ios::out | std::ios::app);
    if (!out.is_open()) {
        return;
    }

    if (generation_idx == 0) {
        out << "generation,fitness,p,i,d,total_frames,target_found_frames,center_hit_frames,laser_hit_frames\n";
    }

    out << (generation_idx + 1) << ","
        << best.fitness << ","
        << best.genome.p << ","
        << best.genome.i << ","
        << best.genome.d << ","
        << best.metrics.total_frames << ","
        << best.metrics.target_found_frames << ","
        << best.metrics.center_hit_frames << ","
        << best.metrics.laser_hit_frames << "\n";
}

} // namespace

int _main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;

    const int frame_width = 640;
    const int frame_height = 640;
    const int tcp_port = 5555;
    const std::string log_dir = "ga_training_logs";

    ensure_dir(log_dir);

    camera::Camera cam(frame_width, frame_height, image::Format::FMT_RGB888);
    display::Display disp;
    peripheral::key::add_default_listener();

    TargetTrackingPipeline pipeline;
    PipelineConfig cfg = pipeline.get_config();
    cfg.fx = 381.625f;
    cfg.fy = 381.625f;
    cfg.cx = static_cast<float>(frame_width) * 0.5f;
    cfg.cy = static_cast<float>(frame_height) * 0.5f;
    cfg.pitch_home = 135.0f;
    cfg.yaw_home = 135.0f;
    cfg.pitch_pwm_zero_angle = 135.0f;
    cfg.yaw_pwm_zero_angle = 135.0f;
    cfg.pid_kp = 2.0f;
    cfg.pid_ki = 0.0f;
    cfg.pid_kd = 0.0f;
    cfg.scan_yaw_freq = 0.08f;
    cfg.scan_pitch_freq = 0.10f;
    cfg.pitch_error_sign = 1.0f;
    cfg.yaw_error_sign = 1.0f;
    cfg.enable_serial = false;
    cfg.draw_overlay = false;
    cfg.print_debug = false;
    pipeline.set_config(cfg);

    TrackerConfig tracker_cfg = pipeline.get_tracker_config();
    tracker_cfg.show_debug_windows = false;
    tracker_cfg.print_debug_info = false;
    pipeline.set_tracker_config(tracker_cfg);

    VisionControlTcpServer tcp_server(tcp_port);
    if (!tcp_server.start()) {
        return -1;
    }

    constexpr int kStreamPort = 8000;
    constexpr uint64_t kStreamIntervalMs = 200;
    const char *kStreamHtml = "<html><body><h1>mygo_ga_pipeline_irl</h1><img src='/stream'></body></html>";
    http::JpegStreamer stream("", kStreamPort);
    stream.set_html(kStreamHtml);
    stream.start();

    std::mt19937 rng(20260411);
    std::vector<Genome> genomes;
    genomes.reserve(kPopulationSize);

    Genome base;
    base.p_min = kPidPMin;
    base.p_max = kPidPMax;
    std::uniform_real_distribution<float> up(0.0f, 1.0f);
    for (int i = 0; i < kPopulationSize; ++i) {
        Genome g = base;
        if (i == 0) {
            // 第一代第一个样本固定为 pipeline_uart 的基准 PID。
            g.p = cfg.pid_kp;
            g.i = cfg.pid_ki;
            g.d = cfg.pid_kd;
        } else {
            g.p = g.p_min + (g.p_max - g.p_min) * up(rng);
            g.i = g.i_min + (g.i_max - g.i_min) * up(rng);
            g.d = g.d_min + (g.d_max - g.d_min) * up(rng);
        }
        g.clamp();
        genomes.push_back(g);
    }

    VisionAppState app_state = VisionAppState::Idle;
    bool recognition_active = false;
    bool training_active = false;
    bool ga_session_armed = false;

    int generation_idx = 0;
    int sample_idx = 0;
    TrainingPhase train_phase = TrainingPhase::Idle;

    std::vector<EvaluatedSample> generation_evaluated;
    generation_evaluated.reserve(kPopulationSize);

    EpisodeMetrics active_metrics;
    int lost_target_streak = 0;
    uint64_t phase_start_ms = time::ticks_ms();

    float startup_pitch_deg = cfg.pitch_home;
    float startup_yaw_deg = cfg.yaw_home;

    TrackState last_state = TrackState::Waiting;
    PipelineOutput last_output;
    last_output.state = TrackState::Waiting;

    uint64_t last_tick_ms = time::ticks_ms();
    uint64_t last_stream_ms = time::ticks_ms();

    auto reset_for_next_sample = [&]() {
        if (sample_idx >= kPopulationSize || generation_idx >= kGenerationCount) {
            return;
        }

        cfg.pitch_home = startup_pitch_deg;
        cfg.yaw_home = startup_yaw_deg;
        cfg.pid_kp = genomes[sample_idx].p;
        cfg.pid_ki = genomes[sample_idx].i;
        cfg.pid_kd = genomes[sample_idx].d;
        pipeline.set_config(cfg);

        pipeline.reset();
        pipeline.set_current_angles(startup_pitch_deg, startup_yaw_deg);
        pipeline.set_control_enabled(true);
        pipeline.start_tracking();

        active_metrics = EpisodeMetrics();
        lost_target_streak = 0;
        train_phase = TrainingPhase::TrackingSample;
        phase_start_ms = time::ticks_ms();

        log::info("[GA] generation=%d sample=%d/%d pid=(%.4f, %.4f, %.4f)",
                  generation_idx + 1,
                  sample_idx + 1,
                  kPopulationSize,
                  cfg.pid_kp,
                  cfg.pid_ki,
                  cfg.pid_kd);
    };

    auto finish_current_sample = [&]() {
        EvaluatedSample s;
        s.genome = genomes[sample_idx];
        s.metrics = active_metrics;
        s.fitness = compute_fitness(active_metrics);
        s.generation = generation_idx;
        s.sample_index = sample_idx;
        generation_evaluated.push_back(s);

        const float mean_center_dist = (active_metrics.target_found_frames > 0)
            ? active_metrics.sum_center_dist_px / static_cast<float>(active_metrics.target_found_frames)
            : 9999.0f;
        log::info("[GA] gen=%d sample=%d fitness=%.4f mean_center_dist=%.2f hit_frames=%d",
                  generation_idx + 1,
                  sample_idx + 1,
                  s.fitness,
                  mean_center_dist,
                  active_metrics.center_hit_frames);

        sample_idx += 1;

        if (sample_idx >= kPopulationSize) {
            std::sort(generation_evaluated.begin(), generation_evaluated.end(),
                      [](const EvaluatedSample &a, const EvaluatedSample &b) {
                          return a.fitness > b.fitness;
                      });

            const EvaluatedSample &best = generation_evaluated.front();
            const float best_mean_center_dist = (best.metrics.target_found_frames > 0)
                ? best.metrics.sum_center_dist_px / static_cast<float>(best.metrics.target_found_frames)
                : 9999.0f;
            const float best_mean_laser_dist = (best.metrics.laser_hit_frames > 0)
                ? best.metrics.sum_laser_target_dist_px / static_cast<float>(best.metrics.laser_hit_frames)
                : 9999.0f;

            log::info("[GA][GEN %d] best fitness=%.4f best_center_dist=%.2f best_laser_dist=%.2f pid=(%.5f,%.5f,%.5f)",
                      generation_idx + 1,
                      best.fitness,
                      best_mean_center_dist,
                      best_mean_laser_dist,
                      best.genome.p,
                      best.genome.i,
                      best.genome.d);

            write_generation_report(log_dir, generation_idx, generation_evaluated);
            append_best_pid(log_dir, generation_idx, best);

            generation_idx += 1;
            if (generation_idx >= kGenerationCount) {
                train_phase = TrainingPhase::Finished;
                training_active = false;
                ga_session_armed = false;
                pipeline.set_control_enabled(false);
                log::info("[GA] training finished. all generations completed.");
                return;
            }

            const float progress = clamp_value(
                static_cast<float>(generation_idx) / std::max(1, kGenerationCount - 1),
                0.0f,
                1.0f);
            const float mutation_rate = kMutationRateStart + (kMutationRateEnd - kMutationRateStart) * progress;
            const float mutation_sigma = kMutationSigmaStart + (kMutationSigmaEnd - kMutationSigmaStart) * progress;
            log::info("[GA] evolve with mutation_rate=%.4f sigma=%.4f", mutation_rate, mutation_sigma);
            genomes = evolve_population(generation_evaluated, rng, kPopulationSize, 2, mutation_rate, mutation_sigma);
            generation_evaluated.clear();
            sample_idx = 0;
        }

        train_phase = TrainingPhase::PrepareSample;
    };

    while (!app::need_exit()) {
        PendingControl control;
        VisionStatusSnapshot snapshot;
        snapshot.app_state = app_state;
        snapshot.tracking_state = recognition_active ? state_to_text(last_output.state) : "Stopped";
        snapshot.active = recognition_active;
        snapshot.training_active = training_active;
        snapshot.can_scan = recognition_active && !training_active;
        snapshot.target_found = recognition_active ? last_output.target_found : false;
        snapshot.laser_found = recognition_active ? last_output.laser_found : false;
        snapshot.target_x = (recognition_active && last_output.target_found) ? static_cast<int>(std::lround(last_output.target_pos.x)) : -1;
        snapshot.target_y = (recognition_active && last_output.target_found) ? static_cast<int>(std::lround(last_output.target_pos.y)) : -1;
        snapshot.command = recognition_active ? last_output.command : "";
        tcp_server.poll(snapshot, control);

        if (control.stop_requested) {
            recognition_active = false;
            training_active = false;
            ga_session_armed = false;
            train_phase = TrainingPhase::Idle;
            app_state = VisionAppState::Stopped;
            pipeline.reset();
            pipeline.set_control_enabled(false);
            last_state = TrackState::Waiting;
            log::info("[GA] stopped by tcp command");
        }

        if (control.recognition_start_requested && !recognition_active) {
            pipeline.reset();
            // A键之前仅识别，不输出云台控制，让手柄链路保持主控。
            pipeline.set_control_enabled(false);
            pipeline.handle_key(' ');
            recognition_active = true;
            app_state = VisionAppState::Running;
            last_state = TrackState::Waiting;
            log::info("[GA] recognition started by tcp command");
        }

        if (control.tracking_start_requested && !training_active) {
            if (!recognition_active) {
                pipeline.reset();
                pipeline.set_control_enabled(true);
                pipeline.handle_key(' ');
                recognition_active = true;
                app_state = VisionAppState::Running;
                last_state = TrackState::Waiting;
                log::info("[GA] recognition auto-started by tracking request");
            }

            if (control.tracking_init_pose_valid) {
                startup_yaw_deg = pwm_to_angle_deg(control.init_yaw_pwm, cfg.yaw_pwm_zero_angle);
                startup_pitch_deg = pwm_to_angle_deg(control.init_pitch_pwm, cfg.pitch_pwm_zero_angle);
            }

            ga_session_armed = true;
            training_active = true;
            generation_idx = 0;
            sample_idx = 0;
            generation_evaluated.clear();
            train_phase = TrainingPhase::PrepareSample;
            log::info("[GA] armed by A-start. startup pose pitch=%.2f yaw=%.2f", startup_pitch_deg, startup_yaw_deg);
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

        if (recognition_active) {
            if (training_active && train_phase == TrainingPhase::PrepareSample) {
                reset_for_next_sample();
            }

            cv::Mat frame;
            maix::image::image2cv(*img, frame, true, true);
            pipeline.set_control_enabled(training_active);
            out = pipeline.process_frame(frame, dt);

            // 与 mygo_pipeline_uart 对齐并增强：训练中一旦不在 Tracking，持续尝试拉回 Tracking。
            if (training_active && out.state != TrackState::Tracking &&
                (out.target_found || train_phase == TrainingPhase::TrackingSample)) {
                pipeline.start_tracking();
                log::info("[GA][AUTO] Re-enter Tracking from %s", state_to_text(out.state));
            }

            if (training_active && train_phase == TrainingPhase::TrackingSample) {
                active_metrics.total_frames += 1;
                if (out.target_found) {
                    lost_target_streak = 0;
                    active_metrics.target_found_frames += 1;
                    const float dx = out.target_pos.x - out.aim_pos.x;
                    const float dy = out.target_pos.y - out.aim_pos.y;
                    const float center_dist = std::sqrt(dx * dx + dy * dy);
                    active_metrics.sum_center_dist_px += center_dist;
                    if (center_dist <= kCenterHitThresholdPx) {
                        active_metrics.center_hit_frames += 1;
                    }
                } else {
                    lost_target_streak += 1;
                    if (lost_target_streak >= kLostReturnToStartFrames) {
                        // 丢失目标后不搜索，直接回起始位并继续追踪。
                        pipeline.set_current_angles(startup_pitch_deg, startup_yaw_deg);
                        pipeline.start_tracking();
                        lost_target_streak = 0;
                        log::info("[GA][RECOVER] target lost, return to startup pose and continue tracking");
                    }
                }

                if (out.target_found && out.laser_found && out.laser_target_error_px <= kLaserHitThresholdPx) {
                    active_metrics.laser_hit_frames += 1;
                    active_metrics.sum_laser_target_dist_px += out.laser_target_error_px;
                }

                const float track_sec = static_cast<float>(now_tick_ms - phase_start_ms) / 1000.0f;
                if (track_sec >= kSampleDurationSec) {
                    finish_current_sample();
                }
            }

            if (out.state != last_state) {
                log::info("[GA][STATE] %s -> %s", state_to_text(last_state), state_to_text(out.state));
                last_state = out.state;
            }
        }

        last_output = out;

        draw_training_overlay(*img,
                              out,
                              generation_idx,
                              std::min(sample_idx, kPopulationSize - 1),
                              train_phase,
                              active_metrics,
                              genomes[std::min(sample_idx, kPopulationSize - 1)]);

        if (recognition_active && (now_tick_ms - last_stream_ms) >= kStreamIntervalMs) {
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

    stream.stop();
    return 0;
}

int main(int argc, char *argv[])
{
    sys::register_default_signal_handle();
    CATCH_EXCEPTION_RUN_RETURN(_main, -1, argc, argv);
}
