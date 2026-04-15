#ifndef TARGET_TRACKING_PIPELINE_HPP
#define TARGET_TRACKING_PIPELINE_HPP

#include <opencv2/opencv.hpp>
#include <string>

#include "TargetTracking/GimbalControl.hpp"
#include "TargetTracking/TargetTracker.hpp"

enum class TrackState { Waiting, Searching, Locked, Tracking };
enum class MicroSimMode { Disabled = 0, SmallFixed = 1, BigVariable = 2 };

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

    // Micro simulation (prediction assist)
    bool enable_micro_sim = false;
    MicroSimMode micro_sim_mode = MicroSimMode::Disabled;
    // Small energy mechanism: fixed speed = pi/3 rad/s
    float micro_sim_small_speed_rad_s = static_cast<float>(CV_PI) / 3.0f;
    // Big energy mechanism: spd = a * sin(omega * t) + b, b = 2.090 - a
    float micro_sim_big_a_min = 0.780f;
    float micro_sim_big_a_max = 1.045f;
    float micro_sim_big_omega_min = 1.884f;
    float micro_sim_big_omega_max = 2.000f;
    bool micro_sim_random_direction = true;
    // Real speed follows target speed with first-order lag, <= 500 ms
    float micro_sim_speed_lag_sec = 0.5f;
    // Max time to trust prediction-only control when vision is missing
    float micro_sim_prediction_hold_sec = 0.45f;
    // Use softer control when only prediction is available
    float micro_sim_pd_gain_scale = 0.65f;
    // 预测相位提前时间（秒），用于有测量时的前馈融合。
    float micro_sim_phase_lead_sec = 0.08f;
    // 有测量时，融合比例：tracking = (1-r)*meas + r*pred_lead
    float micro_sim_blend_ratio = 0.30f;

    // Search scan parameters
    float scan_yaw_amp = 30.0f;
    float scan_pitch_amp = 30.0f;
    float scan_yaw_freq = 0.1f;   // Hz
    float scan_pitch_freq = 0.15f; // Hz
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
    bool target_from_roi = false;
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
    bool sim_active = false;
    bool sim_ready = false;
    bool sim_predicted = false;
    bool sim_lead_used = false;
    cv::Point2f sim_target_pos{-1.0f, -1.0f};
    cv::Point2f sim_lead_pos{-1.0f, -1.0f};
    float sim_angle_rad = 0.0f;
    float sim_speed_rad_s = 0.0f;
    float sim_target_speed_rad_s = 0.0f;
    std::string sim_equation;
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
    /// 外部注入当前姿态，作为追踪初始角度。
    void set_current_angles(float pitch_deg, float yaw_deg);

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
    float pid_step(float error, float dt, PID& pid, float integral_limit, bool allow_integral = true);
    /// 清空PID积分与历史误差。
    void reset_pid(PID& pid);
    /// 把配置中的PID参数写入俯仰/偏航控制器。
    void apply_pid_gains_from_config();
    /// 重置微仿真内部状态并重采样大能量参数。
    void reset_micro_sim_state();
    /// 按配置更新微仿真角速度和角度状态。
    void update_micro_sim_dynamics(float dt);
    /// 用测量到的靶心与目标位置同步仿真轨道参数。
    void sync_micro_sim_orbit_from_measurement(const TargetInfo& info);
    /// 在无测量时给出短时预测目标像素坐标。
    bool get_micro_sim_prediction(float dt, cv::Point2f& out_pos, bool account_missing = true);

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

    bool sim_initialized_ = false;
    float sim_time_ = 0.0f;
    float sim_angle_rad_ = 0.0f;
    float sim_speed_rad_s_ = 0.0f;
    float sim_target_speed_rad_s_ = 0.0f;
    float sim_no_measure_time_ = 0.0f;
    int sim_direction_sign_ = 1;
    float sim_big_a_ = 0.900f;
    float sim_big_omega_ = 1.950f;
    float sim_big_b_ = 1.190f;
    cv::Point2f sim_orbit_center_px_{-1.0f, -1.0f};
    float sim_orbit_radius_px_ = 0.0f;
    bool sim_orbit_valid_ = false;
    std::string sim_equation_text_;

    GimbalControl gimbal_;
    TargetTracker tracker_;
};

#endif // TARGET_TRACKING_PIPELINE_HPP
