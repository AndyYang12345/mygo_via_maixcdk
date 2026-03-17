#ifndef TARGET_TRACKING_PIPELINE_HPP
#define TARGET_TRACKING_PIPELINE_HPP

#include <opencv2/opencv.hpp>
#include <string>

#include "TargetTracking/GimbalControl.hpp"
#include "TargetTracking/TargetTracker.hpp"

enum class TrackState { Waiting, Searching, Locked, Tracking };

struct PipelineConfig {
    // Camera intrinsics (set <0 to auto from frame size)
    float fx = -1.0f;
    float fy = -1.0f;
    float cx = -1.0f;
    float cy = -1.0f;

    // Home angles
    float pitch_home = 60.0f;
    float yaw_home = 105.0f;
    float pitch_min = 30.0f;
    float pitch_max = 90.0f;
    float yaw_min = 0.0f;
    float yaw_max = 270.0f;

    // PID limits
    float max_speed = 180.0f;
    float integral_limit = 30.0f;

    // Search/lock behavior
    int lock_required = 30;
    int lost_required = 10;

    // Search scan parameters
    float scan_yaw_amp = 30.0f;
    float scan_pitch_amp = 15.0f;
    float scan_yaw_freq = 0.15f;   // Hz
    float scan_pitch_freq = 0.10f; // Hz
    float scan_phase = static_cast<float>(CV_PI) * 0.5f;

    // Output behavior
    bool draw_overlay = true;
    bool print_debug = false;

    // Serial
    bool enable_serial = false;
    std::string serial_device = "/dev/ttyUSB0";
    int serial_baud = 115200;
};

struct PipelineOutput {
    cv::Mat canvas;
    std::string command;
    TrackState state = TrackState::Searching;
    int lock_count = 0;
    int lost_count = 0;
    bool target_found = false;
    cv::Point2f target_pos{-1.0f, -1.0f};
    bool laser_found = false;
    cv::Point2f laser_pos{-1.0f, -1.0f};
    float laser_target_error_px = 0.0f;
    float pitch_angle = 0.0f;
    float yaw_angle = 0.0f;
    float pitch_speed = 0.0f;
    float yaw_speed = 0.0f;
};

class TargetTrackingPipeline {
public:
    TargetTrackingPipeline();

    void set_config(const PipelineConfig& config);
    PipelineConfig get_config() const;

    void set_tracker_config(const TrackerConfig& config);
    TrackerConfig get_tracker_config() const;

    void reset();
    void handle_key(int key);

    PipelineOutput process_frame(const cv::Mat& frame, float dt);

    float get_pitch_angle() const;
    float get_yaw_angle() const;

    bool open_serial();
    void close_serial();
    bool is_serial_open() const;
    bool send_raw_serial_command(const std::string& command);

private:
    struct PID {
        float kp{11.8354f};
        float ki{0.315478f};
        float kd{0.0215511f};
        float integral{0.0f};
        float prev_error{0.0f};
        bool has_prev{false};
    };

    float clamp_value(float v, float lo, float hi) const;
    float pid_step(float error, float dt, PID& pid, float integral_limit);
    void reset_pid(PID& pid);

    PipelineConfig config_;
    TrackState state_ = TrackState::Searching;
    int lock_count_ = 0;
    int lost_count_ = 0;
    float scan_time_ = 0.0f;

    float pitch_angle_ = 0.0f;
    float yaw_angle_ = 0.0f;
    float pitch_speed_ = 0.0f;
    float yaw_speed_ = 0.0f;

    PID pid_pitch_;
    PID pid_yaw_;

    GimbalControl gimbal_;
    TargetTracker tracker_;
};

#endif // TARGET_TRACKING_PIPELINE_HPP
