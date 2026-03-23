#include "maix_basic.hpp"
#include "maix_camera.hpp"
#include "maix_display.hpp"
#include "maix_image_cv.hpp"
#include "maix_pinmap.hpp"
#include "main.h"

#include "TargetTracking/TargetTrackingPipeline.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <string>

using namespace maix;

namespace {

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

void draw_pipeline_overlay(image::Image &img, const PipelineOutput &out)
{
    const image::Color target_color = image::Color::from_rgb(255, 64, 64);
    const image::Color laser_color = image::Color::from_rgb(255, 255, 0);
    const image::Color info_color = image::Color::from_rgb(0, 255, 0);
    const image::Color ok_color = image::Color::from_rgb(0, 255, 0);
    const image::Color bad_color = image::Color::from_rgb(255, 64, 64);
    const image::Color hit_color = image::Color::from_rgb(0, 255, 0);
    const image::Color miss_color = image::Color::from_rgb(255, 180, 0);
    constexpr float kHitThresholdPx = 30.0f;

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
        img.draw_string(6, 50, line3, detect_hit ? hit_color : miss_color, 1.2f);
    } else {
        img.draw_string(6, 50, "detect:N/A", miss_color, 1.2f);
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

    std::string line1 = std::string("state:") + state_to_text(out.state) +
                        " lock:" + std::to_string(out.lock_count) +
                        " lost:" + std::to_string(out.lost_count);
    img.draw_string(6, 6, line1, info_color, 1.2f);

    char line2[128] = {0};
    snprintf(line2,
             sizeof(line2),
             "pitch:%.2f yaw:%.2f",
             out.pitch_angle,
             out.yaw_angle);
    img.draw_string(6, 28, line2, info_color, 1.2f);

    std::string target_status = std::string("target:") + (out.target_found ? "FOUND" : "NOT FOUND");
    std::string laser_status = std::string("laser:") + (out.laser_found ? "FOUND" : "NOT FOUND");
    img.draw_string(6, 72, target_status, out.target_found ? ok_color : bad_color, 1.2f);
    img.draw_string(6, 94, laser_status, out.laser_found ? ok_color : bad_color, 1.2f);
    img.draw_string(6, 116, "auto scan->track | q:quit", info_color, 1.2f);
}

} // namespace

int _main(int argc, char *argv[])
{
    int frame_width = 640;
    int frame_height = 480;
    bool enable_pipeline_uart = true;
    int uart_baud = 115200;
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
        } else if (arg == "--baud" && i + 1 < argc) {
            uart_baud = std::stoi(argv[++i]);
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

    if (enable_pipeline_uart) {
        setup_uart_pinmux(device_name, uart_device);
    }

    const bool auto_start_search = true;
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
        // if (opened) {
        //     const std::string init_arm_cmd = "{#00P1500T1000P1350T1000P2300T1000P1500T1000P1500T1000}";
        //     bool init_ok = pipeline.send_raw_serial_command(init_arm_cmd);
        //     log::info("arm init cmd sent: %s", init_ok ? "true" : "false");
        // }
    }

    if (auto_start_search) {
        pipeline.handle_key(' ');
        log::info("[AUTO] Waiting -> Searching (startup)");
    }

    camera::Camera cam(frame_width, frame_height, image::Format::FMT_RGB888);
    display::Display disp;

    uint64_t last_tick_ms = time::ticks_ms();
    TrackState last_state = TrackState::Waiting;
    while (!app::need_exit()) {
        image::Image *img = cam.read();
        if (!img) {
            time::sleep_ms(5);
            continue;
        }

        uint64_t now_tick_ms = time::ticks_ms();
        float dt = static_cast<float>(now_tick_ms - last_tick_ms) / 1000.0f;
        last_tick_ms = now_tick_ms;
        dt = clamp_value(dt, 0.001f, 0.05f);

        cv::Mat frame;
        maix::image::image2cv(*img, frame, true, true);
        PipelineOutput out = pipeline.process_frame(frame, dt);

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

        draw_pipeline_overlay(*img, out);
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
