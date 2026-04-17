#ifndef TARGET_TRACKING_PIPELINE_HPP
#define TARGET_TRACKING_PIPELINE_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <utility>

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
    bool enable_view_angle_feedforward = false;

    // Debug open-loop orbit mode in Tracking state:
    // use phase(t)=phase0+omega*t to generate servo angles without visual PID.
    bool enable_open_loop_phase_orbit = false;
    float open_loop_omega_rad_s = 1.0471976f;      // default: pi/3 rad/s
    float open_loop_phase_init_rad = 0.0f;
    float open_loop_default_distance_mm = 800.0f;  // fallback when no valid estimate

    // Output-angle smoothing for trajectory generation.
    // 1) first-order interpolation with speed limit, 2) first-order low-pass.
    bool enable_angle_interpolation = true;
    float angle_interp_max_speed_deg_s = 220.0f;
    bool enable_angle_lowpass = true;
    float angle_lowpass_tau_s = 0.06f;

    // Closed-loop phase lock on top of open-loop orbit.
    // Keep omega feedforward as primary driver, and map phase error to temporary omega increment.
    bool enable_phase_lock = false;
    float phase_lock_kp = 0.30f;
    float phase_lock_ki = 0.06f;
    float phase_lock_kd = 0.00f;
    float phase_lock_max_step_rad = 0.02f;
    float phase_lock_integral_limit = 1.2f;
    float phase_lock_omega_bias_limit_rad_s = 0.35f;
    float phase_lock_innovation_gate_rad = 3.1415926f;
    int phase_lock_outlier_freeze_frames = 2;
    int phase_lock_min_valid_frames = 1;

    // ROI-based speed identification in Searching state.
    bool enable_speed_identification = false;
    float speed_id_warmup_s = 0.30f;
    float speed_id_validate_s = 1.50f;
    int speed_id_min_samples = 20;
    float speed_id_omega_smooth_alpha = 0.20f;
    float speed_id_radius_smooth_alpha = 0.15f;
    float speed_id_roi_tolerance_ratio = 0.45f;
    float speed_id_min_tolerance_px = 12.0f;

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
    bool tracking_recovery_requested = false;
    bool target_found = false;
    cv::Point2f target_pos{-1.0f, -1.0f};
    cv::Point2f board_pos{-1.0f, -1.0f};
    bool roi_active = false;
    cv::Rect roi_rect{-1, -1, 0, 0};
    bool laser_found = false;
    cv::Point2f laser_pos{-1.0f, -1.0f};
    cv::Point2f aim_pos{-1.0f, -1.0f};
    bool aim_from_laser = false;
    float laser_target_error_px = 0.0f;
    float board_distance_mm = -1.0f;
    bool view_angle_valid = false;
    float view_delta_yaw_rad = 0.0f;
    float view_delta_pitch_rad = 0.0f;
    float feedforward_pitch_angle = 0.0f;
    float feedforward_yaw_angle = 0.0f;
    bool open_loop_active = false;
    float open_loop_phase_rad = 0.0f;
    float open_loop_omega_rad_s = 0.0f;
    float open_loop_distance_mm = -1.0f;
    bool phase_lock_active = false;
    float phase_lock_target_phase_rad = 0.0f;
    float phase_lock_error_rad = 0.0f;
    float phase_lock_step_rad = 0.0f;
    float phase_lock_omega_bias_rad_s = 0.0f;
    bool phase_lock_skipped = false;
    int phase_lock_outlier_count = 0;
    bool predicted_pos_valid = false;
    cv::Point2f predicted_pos{-1.0f, -1.0f};
    bool speed_identifying = false;
    bool speed_identified = false;
    bool speed_identified_event = false;
    float identified_omega_rad_s = 0.0f;
    float identified_phase_rad = 0.0f;
    float instant_omega_rad_s = 0.0f;
    float fitted_omega_rad_s = 0.0f;
    int speed_fit_samples = 0;
    float speed_validation_error_px = -1.0f;
    float speed_validation_tolerance_px = -1.0f;
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
    /// 外部注入当前姿态，作为追踪初始角度。
    void set_current_angles(float pitch_deg, float yaw_deg);

    /// 重置状态机、PID和扫描时间。
    void reset();
    /// 处理按键事件（例如空格触发状态切换）。
    void handle_key(int key);

    /// 处理单帧输入并输出控制命令与可视化结果。
    PipelineOutput process_frame(const cv::Mat& frame, float dt);

    /// 将当前舵机角度与视角偏移合成新的舵机角度（deg）。
    std::pair<float, float> compute_servo_angles_from_offsets(float current_pitch_deg,
                                                              float current_yaw_deg,
                                                              float yaw_offset_rad,
                                                              float pitch_offset_rad) const;

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

    bool open_loop_phase_active_ = false;
    float open_loop_phase_rad_ = 0.0f;
    float open_loop_base_pitch_deg_ = 0.0f;
    float open_loop_base_yaw_deg_ = 0.0f;
    float open_loop_distance_mm_ = -1.0f;
    float open_loop_locked_omega_rad_s_ = 0.0f;
    bool open_loop_locked_from_fit_ = false;
    float phase_lock_integral_ = 0.0f;
    float phase_lock_omega_bias_rad_s_ = 0.0f;
    float phase_lock_prev_error_rad_ = 0.0f;
    bool phase_lock_has_prev_error_ = false;
    int phase_lock_outlier_count_ = 0;
    int phase_lock_freeze_left_ = 0;
    int phase_lock_valid_frames_ = 0;
    float last_board_distance_mm_ = -1.0f;

    bool speed_id_active_ = false;
    bool speed_id_valid_ = false;
    bool speed_id_event_pending_ = false;
    int speed_id_samples_ = 0;
    bool speed_id_all_in_roi_ = true;
    float speed_id_warmup_elapsed_s_ = 0.0f;
    float speed_id_validate_elapsed_s_ = 0.0f;
    float speed_id_phase_init_rad_ = 0.0f;
    float speed_id_last_phase_rad_ = 0.0f;
    float speed_id_unwrapped_phase_rad_ = 0.0f;
    float speed_id_unwrapped_phase_start_rad_ = 0.0f;
    float speed_id_omega_rad_s_ = 0.0f;
    float speed_id_inst_omega_rad_s_ = 0.0f;
    float speed_id_radius_px_ = 0.0f;
    bool speed_id_validation_started_ = false;
    bool speed_id_has_prev_vec_ = false;
    cv::Point2f speed_id_prev_vec_{0.0f, 0.0f};
    float speed_id_omega_sum_rad_s_ = 0.0f;
    int speed_id_omega_count_ = 0;
    float speed_id_reg_sum_t_ = 0.0f;
    float speed_id_reg_sum_p_ = 0.0f;
    float speed_id_reg_sum_tt_ = 0.0f;
    float speed_id_reg_sum_tp_ = 0.0f;
    float speed_id_last_error_px_ = -1.0f;
    float speed_id_last_tolerance_px_ = -1.0f;
    cv::Point2f speed_id_last_predicted_pos_{-1.0f, -1.0f};

    GimbalControl gimbal_;
    TargetTracker tracker_;
};

#endif // TARGET_TRACKING_PIPELINE_HPP
