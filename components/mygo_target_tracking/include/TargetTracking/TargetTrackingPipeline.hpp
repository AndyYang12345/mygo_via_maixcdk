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
    float pitch_home = 135.0f;
    float yaw_home = 135.0f;

    // Servo PWM zero-angle calibration (angle that maps to PWM=1500)
    float pitch_pwm_zero_angle = 135.0f;
    float yaw_pwm_zero_angle = 135.0f;

    // PID limits
    float max_speed = 180.0f;
    float integral_limit = 30.0f;
    float pid_kp = 11.8354f;
    float pid_ki = 0.315478f;
    float pid_kd = 0.0215511f;

    // Control direction (set to -1.0f when axis direction is reversed)
    float pitch_error_sign = -1.0f;
    float yaw_error_sign = 1.0f;

    // Search/lock behavior
    int lock_required = 30;
    int lost_required = 10;

    // Search scan parameters
    float scan_yaw_amp = 60.0f;
    float scan_pitch_amp = 30.0f;
    float scan_yaw_freq = 0.15f;   // Hz
    float scan_pitch_freq = 0.10f; // Hz
    // yaw 默认从中心起扫（1500PWM 对应零位），随后左右震荡
    float scan_yaw_phase = 0.0f;
    // pitch 默认从上方向下扫（围绕零位/1500PWM往返）
    float scan_pitch_phase = 0.0f;

    // Output behavior
    bool draw_overlay = true;
    bool print_debug = false;
    bool control_enabled = true;

    // Serial
    bool enable_serial = false;
    std::string serial_device = "/dev/ttyUSB0";
    int serial_baud = 115200;
};

struct PipelineOutput {
    cv::Mat canvas;
    std::string command;
    TrackState state = TrackState::Waiting;
    int lock_count = 0;
    int lost_count = 0;
    bool target_found = false;
    cv::Point2f target_pos{-1.0f, -1.0f};
    bool roi_active = false;
    cv::Rect roi_rect{-1, -1, 0, 0};
    bool laser_found = false;
    cv::Point2f laser_pos{-1.0f, -1.0f};
    cv::Point2f aim_pos{-1.0f, -1.0f};
    bool aim_from_laser = false;
    float laser_target_error_px = 0.0f;
    float pitch_angle = 0.0f;
    float yaw_angle = 0.0f;
    float pitch_speed = 0.0f;
    float yaw_speed = 0.0f;
};

class TargetTrackingPipeline {
public:
    /// 构造管线并加载默认参数到控制器与PID。
    TargetTrackingPipeline();

    /// 设置整条管线配置（控制、扫描、串口等）。
    void set_config(const PipelineConfig& config);
    /// 获取当前管线配置快照。
    PipelineConfig get_config() const;

    /// 设置目标检测器配置。
    void set_tracker_config(const TrackerConfig& config);
    /// 获取目标检测器配置。
    TrackerConfig get_tracker_config() const;

    /// 设置控制使能。
    void set_control_enabled(bool enabled);
    /// 显式切换到追踪状态。
    void start_tracking();

    /// 重置状态机、PID和扫描时间。
    void reset();
    /// 处理按键事件（例如空格触发状态切换）。
    void handle_key(int key);

    /// 处理单帧输入并输出控制命令与可视化结果。
    PipelineOutput process_frame(const cv::Mat& frame, float dt);

    /// 读取当前俯仰角状态。
    float get_pitch_angle() const;
    /// 读取当前偏航角状态。
    float get_yaw_angle() const;

    /// 根据配置尝试打开串口。
    bool open_serial();
    /// 关闭串口连接。
    void close_serial();
    /// 查询串口是否已打开。
    bool is_serial_open() const;
    /// 直接发送原始串口命令（绕过自动生成）。
    bool send_raw_serial_command(const std::string& command);

private:
    struct PID {
        float kp{0.0f};
        float ki{0.0f};
        float kd{0.0f};
        float integral{0.0f};
        float prev_error{0.0f};
        bool has_prev{false};
    };

    /// 将数值限制在区间 [lo, hi] 内。
    float clamp_value(float v, float lo, float hi) const;
    /// 计算单步PID输出速度命令。
    float pid_step(float error, float dt, PID& pid, float integral_limit);
    /// 清空PID积分与历史误差。
    void reset_pid(PID& pid);
    /// 把配置中的PID参数写入俯仰/偏航控制器。
    void apply_pid_gains_from_config();

    PipelineConfig config_;
    bool control_enabled_ = true;
    TrackState state_ = TrackState::Waiting;
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
