#include "TargetTracking/TargetTrackingPipeline.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>

TargetTrackingPipeline::TargetTrackingPipeline() {
    control_enabled_ = config_.control_enabled;
    gimbal_.set_pitch_zero_angle_deg(config_.pitch_pwm_zero_angle);
    gimbal_.set_yaw_zero_angle_deg(config_.yaw_pwm_zero_angle);
    apply_pid_gains_from_config();
    pitch_angle_ = config_.pitch_home;
    yaw_angle_ = config_.yaw_home;
    reset_micro_sim_state();
}

void TargetTrackingPipeline::set_config(const PipelineConfig& config) {
    config_ = config;
    control_enabled_ = config_.control_enabled;
    gimbal_.set_pitch_zero_angle_deg(config_.pitch_pwm_zero_angle);
    gimbal_.set_yaw_zero_angle_deg(config_.yaw_pwm_zero_angle);
    apply_pid_gains_from_config();
    pitch_angle_ = config_.pitch_home;
    yaw_angle_ = config_.yaw_home;
    reset_micro_sim_state();
}

PipelineConfig TargetTrackingPipeline::get_config() const {
    return config_;
}

void TargetTrackingPipeline::set_tracker_config(const TrackerConfig& config) {
    tracker_.set_config(config);
}

TrackerConfig TargetTrackingPipeline::get_tracker_config() const {
    return tracker_.get_config();
}

void TargetTrackingPipeline::set_control_enabled(bool enabled) {
    control_enabled_ = enabled;
    config_.control_enabled = enabled;
    if (!enabled) {
        pitch_speed_ = 0.0f;
        yaw_speed_ = 0.0f;
        reset_pid(pid_pitch_);
        reset_pid(pid_yaw_);
    }
}

void TargetTrackingPipeline::start_tracking() {
    state_ = TrackState::Tracking;
    lock_count_ = 0;
    lost_count_ = 0;
    sim_no_measure_time_ = 0.0f;
    reset_pid(pid_pitch_);
    reset_pid(pid_yaw_);
}

void TargetTrackingPipeline::set_current_angles(float pitch_deg, float yaw_deg) {
    pitch_angle_ = clamp_value(pitch_deg, 0.0f, 270.0f);
    yaw_angle_ = clamp_value(yaw_deg, 0.0f, 270.0f);
    pitch_speed_ = 0.0f;
    yaw_speed_ = 0.0f;
}

void TargetTrackingPipeline::reset() {
    state_ = TrackState::Waiting;
    lock_count_ = 0;
    lost_count_ = 0;
    scan_time_ = 0.0f;
    pitch_angle_ = config_.pitch_home;
    yaw_angle_ = config_.yaw_home;
    pitch_speed_ = 0.0f;
    yaw_speed_ = 0.0f;
    reset_pid(pid_pitch_);
    reset_pid(pid_yaw_);
    tracker_.reset_roi_tracking();
    reset_micro_sim_state();
}

void TargetTrackingPipeline::handle_key(int key) {
    if (key == ' ') {
        if (state_ == TrackState::Waiting) {
            state_ = TrackState::Searching;
            scan_time_ = 0.0f;
            lock_count_ = 0;
            lost_count_ = 0;
        } else if (state_ == TrackState::Locked) {
            state_ = TrackState::Tracking;
            reset_pid(pid_pitch_);
            reset_pid(pid_yaw_);
            lost_count_ = 0;
        }
    }
}

PipelineOutput TargetTrackingPipeline::process_frame(const cv::Mat& frame, float dt) {
    PipelineOutput output;
    output.state = state_;
    output.lock_count = lock_count_;
    output.lost_count = lost_count_;

    if (frame.empty()) {
        return output;
    }

    dt = clamp_value(dt, 0.001f, 0.05f);

    float fx = (config_.fx > 0.0f) ? config_.fx : frame.cols * 0.6f;
    float fy = (config_.fy > 0.0f) ? config_.fy : frame.rows * 0.6f;
    float cx = (config_.cx >= 0.0f) ? config_.cx : frame.cols * 0.5f;
    float cy = (config_.cy >= 0.0f) ? config_.cy : frame.rows * 0.5f;

    TargetInfo info = tracker_.process_frame(frame);
    const bool has_target = info.found;
    const bool has_laser = info.laser_found;

    if (has_target) {
        sync_micro_sim_orbit_from_measurement(info);
        sim_no_measure_time_ = 0.0f;
    }
    update_micro_sim_dynamics(dt);

    cv::Point2f target_pos(-1.0f, -1.0f);
    if (has_target) {
        target_pos = info.target_center;
    }
    output.roi_active = info.roi_active;
    output.roi_rect = info.roi_rect;
    cv::Point2f laser_pos(-1.0f, -1.0f);
    if (has_laser) {
        laser_pos = info.laser_center;
    }
    output.aim_pos = cv::Point2f(cx, cy);
    output.aim_from_laser = false;
    output.sim_active = config_.enable_micro_sim && config_.micro_sim_mode != MicroSimMode::Disabled;
    output.sim_ready = output.sim_active && sim_initialized_ && sim_orbit_valid_;
    output.sim_angle_rad = sim_angle_rad_;
    output.sim_speed_rad_s = sim_speed_rad_s_;
    output.sim_target_speed_rad_s = sim_target_speed_rad_s_;
    output.sim_equation = sim_equation_text_;

    if (output.sim_ready) {
        cv::Point2f sim_preview_pos(-1.0f, -1.0f);
        const float preview_lead_sec = std::max(0.0f, config_.micro_sim_phase_lead_sec);
        if (get_micro_sim_prediction(preview_lead_sec, sim_preview_pos, false)) {
            output.sim_target_pos = sim_preview_pos;
            output.sim_predicted = true;
        }
    }

    if (state_ == TrackState::Waiting) {
        pitch_angle_ = config_.pitch_home;
        yaw_angle_ = config_.yaw_home;
        pitch_speed_ = 0.0f;
        yaw_speed_ = 0.0f;
    } else if (state_ == TrackState::Searching) {
        scan_time_ += dt;
        // Yaw 三角扫描（边界自适应）：避免 home±amp 超出机械范围后被夹平。
        const float yaw_freq = std::max(1e-3f, config_.scan_yaw_freq);
        float yaw_phase01 = std::fmod(scan_time_ * yaw_freq +
                                      config_.scan_yaw_phase / (2.0f * static_cast<float>(CV_PI)),
                                      1.0f);
        if (yaw_phase01 < 0.0f) {
            yaw_phase01 += 1.0f;
        }

        const float yaw_amp = std::max(0.0f, config_.scan_yaw_amp);
        const float yaw_left = std::min(yaw_amp, std::max(0.0f, config_.yaw_home - 0.0f));
        const float yaw_right = std::min(yaw_amp, std::max(0.0f, 270.0f - config_.yaw_home));
        const float yaw_min = config_.yaw_home - yaw_left;
        const float yaw_max = config_.yaw_home + yaw_right;
        const float yaw_span = std::max(0.0f, yaw_max - yaw_min);

        // t=0 从 home 出发优先向左；随后在 [yaw_min, yaw_max] 匀速往返。
        const float tri_phase = std::fmod(yaw_phase01 + 0.25f, 1.0f);
        if (yaw_span < 1e-4f) {
            yaw_angle_ = config_.yaw_home;
            yaw_speed_ = 0.0f;
        } else {
            const float tri01 = 1.0f - std::abs(2.0f * tri_phase - 1.0f); // [0,1]
            yaw_angle_ = yaw_min + yaw_span * tri01;
            const float sweep_speed = 2.0f * yaw_span * yaw_freq;
            yaw_speed_ = (tri_phase < 0.5f ? -sweep_speed : sweep_speed);
        }

        // Pitch保持0相位起始：从中心开始向下做正弦震荡。
        const float pitch_phase = 2.0f * static_cast<float>(CV_PI) * config_.scan_pitch_freq * scan_time_;
        pitch_angle_ = config_.pitch_home + config_.scan_pitch_amp * std::sin(pitch_phase);
        pitch_speed_ = config_.scan_pitch_amp * 2.0f * static_cast<float>(CV_PI) *
                       config_.scan_pitch_freq * std::cos(pitch_phase);

        if (has_target) {
            lock_count_++;
        } else {
            lock_count_ = 0;
        }

        if (lock_count_ >= config_.lock_required) {
            state_ = TrackState::Locked;
            lock_count_ = 0;
            lost_count_ = 0;
        }
    } else if (state_ == TrackState::Locked) {
        pitch_speed_ = 0.0f;
        yaw_speed_ = 0.0f;
        if (has_target) {
            lost_count_ = 0;
        } else {
            lost_count_++;
            if (lost_count_ >= config_.lost_required) {
                state_ = TrackState::Waiting;
                lost_count_ = 0;
                lock_count_ = 0;
                scan_time_ = 0.0f;
                pitch_angle_ = config_.pitch_home;
                yaw_angle_ = config_.yaw_home;
                pitch_speed_ = 0.0f;
                yaw_speed_ = 0.0f;
                reset_pid(pid_pitch_);
                reset_pid(pid_yaw_);
                tracker_.reset_roi_tracking();
            }
        }
    } else if (state_ == TrackState::Tracking) {
        bool sim_predicted = false;
        cv::Point2f tracking_pos = target_pos;
        cv::Point2f sim_lead_pos(-1.0f, -1.0f);
        bool sim_lead_ok = false;

        if (has_target) {
            const float lead_t = std::max(0.0f, config_.micro_sim_phase_lead_sec);
            sim_lead_ok = get_micro_sim_prediction(lead_t, sim_lead_pos, false);
            if (sim_lead_ok) {
                const float r = clamp_value(config_.micro_sim_blend_ratio, 0.0f, 1.0f);
                tracking_pos.x = (1.0f - r) * target_pos.x + r * sim_lead_pos.x;
                tracking_pos.y = (1.0f - r) * target_pos.y + r * sim_lead_pos.y;
                output.sim_lead_used = true;
                output.sim_lead_pos = sim_lead_pos;
                output.sim_target_pos = sim_lead_pos;
            }
        }

        if (!has_target) {
            sim_predicted = get_micro_sim_prediction(dt, tracking_pos);
            if (sim_predicted) {
                output.sim_target_pos = tracking_pos;
                output.sim_predicted = true;
            }
        }

        if (!control_enabled_) {
            pitch_speed_ = 0.0f;
            yaw_speed_ = 0.0f;
        } else if (has_target || sim_predicted) {
            float dx = 0.0f;
            float dy = 0.0f;
            dx = tracking_pos.x - cx;
            dy = tracking_pos.y - cy;
            output.aim_pos = cv::Point2f(cx, cy);
            output.aim_from_laser = false;
            // 使用角度误差驱动PID，避免归一化误差量级过小导致控制输出不足。
            // 通过 atan2(error_px, focal_px) 计算视线偏角（单位：deg）。
            float pitch_error = config_.pitch_error_sign *
                                ( - std::atan2(dy, fy) * 180.0f / static_cast<float>(CV_PI));
            float yaw_error = config_.yaw_error_sign *
                              ( - std::atan2(dx, fx) * 180.0f / static_cast<float>(CV_PI));

            if (config_.print_debug) {
                std::cout << std::fixed << std::setprecision(2)
                          << "Target(px):(" << tracking_pos.x << "," << tracking_pos.y << ")"
                          << " Laser(px):(" << laser_pos.x << "," << laser_pos.y << ")"
                          << " dx,dy:(" << dx << "," << dy << ")"
                          << " pitch_err:" << pitch_error
                          << " yaw_err:" << yaw_error
                          << " sim_pred:" << (sim_predicted ? 1 : 0)
                          << " sim_lead:" << (sim_lead_ok ? 1 : 0)
                          << " cur_pitch:" << pitch_angle_
                          << " cur_yaw:" << yaw_angle_
                          << std::endl;
            }

            const bool allow_integral = !sim_predicted;
            const float gain_scale = sim_predicted
                                        ? clamp_value(config_.micro_sim_pd_gain_scale, 0.0f, 1.0f)
                                        : 1.0f;
            float pitch_speed_cmd = pid_step(pitch_error, dt, pid_pitch_, config_.integral_limit, allow_integral);
            float yaw_speed_cmd = pid_step(yaw_error, dt, pid_yaw_, config_.integral_limit, allow_integral);
            pitch_speed_cmd *= gain_scale;
            yaw_speed_cmd *= gain_scale;

            pitch_speed_ = clamp_value(pitch_speed_cmd, -config_.max_speed, config_.max_speed);
            yaw_speed_ = clamp_value(yaw_speed_cmd, -config_.max_speed, config_.max_speed);

            pitch_angle_ += pitch_speed_ * dt;
            yaw_angle_ += yaw_speed_ * dt;
            pitch_angle_ = clamp_value(pitch_angle_, 0.0f, 270.0f);
            yaw_angle_ = clamp_value(yaw_angle_, 0.0f, 270.0f);

            if (has_target) {
                lost_count_ = 0;
            } else {
                lost_count_ = std::min(lost_count_ + 1, config_.lost_required);
            }
        } else {
            pitch_speed_ = 0.0f;
            yaw_speed_ = 0.0f;
            lost_count_++;
            if (lost_count_ >= config_.lost_required) {
                state_ = TrackState::Waiting;
                lost_count_ = 0;
                lock_count_ = 0;
                scan_time_ = 0.0f;
                pitch_angle_ = config_.pitch_home;
                yaw_angle_ = config_.yaw_home;
                reset_pid(pid_pitch_);
                reset_pid(pid_yaw_);
                tracker_.reset_roi_tracking();
            }
        }
    }

    gimbal_.set_pitch_angle(pitch_angle_);
    gimbal_.set_yaw_angle(yaw_angle_);
    gimbal_.set_pitch_speed(std::abs(pitch_speed_));
    gimbal_.set_yaw_speed(std::abs(yaw_speed_));
    std::string cmd;
    if (control_enabled_) {
        gimbal_.get_command();
        cmd = gimbal_.get_command_buffer();
        if (config_.enable_serial && gimbal_.is_serial_open()) {
            gimbal_.send_command();
        }
    }

    output.state = state_;
    output.lock_count = lock_count_;
    output.lost_count = lost_count_;
    output.command = cmd;
    output.pitch_angle = pitch_angle_;
    output.yaw_angle = yaw_angle_;
    output.pitch_speed = pitch_speed_;
    output.yaw_speed = yaw_speed_;
    output.target_found = has_target;
    output.target_from_roi = has_target && info.target_from_roi;
    output.target_pos = target_pos;
    output.roi_active = info.roi_active;
    output.roi_rect = info.roi_rect;
    output.laser_found = has_laser;
    output.laser_pos = laser_pos;
    output.laser_target_error_px = (has_target && has_laser) ? cv::norm(target_pos - laser_pos) : 0.0f;
    if (has_target) {
        output.sim_target_pos = target_pos;
    }

    if (config_.draw_overlay) {
        output.canvas = frame.clone();

        std::ostringstream status;
        status << std::fixed << std::setprecision(2)
               << "Pitch: " << pitch_angle_ << " deg  |  Yaw: " << yaw_angle_ << " deg";
        cv::putText(output.canvas, status.str(), {20, 40}, cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(220, 220, 220), 2);

        std::ostringstream speed_line;
        speed_line << std::fixed << std::setprecision(2)
                   << "Pitch Speed: " << pitch_speed_ << " deg/s  |  "
                   << "Yaw Speed: " << yaw_speed_ << " deg/s";
        cv::putText(output.canvas, speed_line.str(), {20, 80}, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(180, 220, 255), 2);

        cv::putText(output.canvas, "Serial Cmd:", {20, 130}, cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(255, 255, 0), 2);
        cv::putText(output.canvas, cmd, {20, 170}, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(255, 255, 0), 2);

        std::ostringstream mode_line;
        switch (state_) {
            case TrackState::Waiting:
                mode_line << "Waiting | home position";
                break;
            case TrackState::Searching:
                mode_line << "Searching | lock:" << lock_count_ << "/" << config_.lock_required;
                break;
            case TrackState::Locked:
                mode_line << "Target locked | press SPACE to start tracking";
                break;
            case TrackState::Tracking:
                mode_line << "Tracking(laser-hit) | lost:" << lost_count_ << "/" << config_.lost_required;
                break;
        }
        mode_line << " | q/ESC=quit";
        cv::putText(output.canvas, mode_line.str(), {20, 430}, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(180, 180, 180), 1);

        if (has_target && target_pos.x >= 0.0f) {
            cv::circle(output.canvas, cv::Point(static_cast<int>(target_pos.x), static_cast<int>(target_pos.y)),
                       6, cv::Scalar(0, 0, 255), -1);
        }
        if (output.sim_predicted && output.sim_target_pos.x >= 0.0f) {
            cv::circle(output.canvas,
                       cv::Point(static_cast<int>(output.sim_target_pos.x), static_cast<int>(output.sim_target_pos.y)),
                       5,
                       cv::Scalar(255, 0, 255),
                       2);
            cv::putText(output.canvas,
                        "SIM",
                        cv::Point(static_cast<int>(output.sim_target_pos.x) + 6,
                                  static_cast<int>(output.sim_target_pos.y) - 6),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.45,
                        cv::Scalar(255, 0, 255),
                        1);
        }
        if (output.sim_lead_used && output.sim_lead_pos.x >= 0.0f) {
            cv::circle(output.canvas,
                       cv::Point(static_cast<int>(output.sim_lead_pos.x), static_cast<int>(output.sim_lead_pos.y)),
                       4,
                       cv::Scalar(0, 165, 255),
                       2);
            cv::putText(output.canvas,
                        "LEAD",
                        cv::Point(static_cast<int>(output.sim_lead_pos.x) + 6,
                                  static_cast<int>(output.sim_lead_pos.y) + 12),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.40,
                        cv::Scalar(0, 165, 255),
                        1);
        }
        if (!output.sim_equation.empty()) {
            cv::putText(output.canvas,
                        output.sim_equation,
                        {20, 245},
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.50,
                        cv::Scalar(210, 180, 255),
                        1);
        }
        if (has_laser && laser_pos.x >= 0.0f) {
            cv::circle(output.canvas, cv::Point(static_cast<int>(laser_pos.x), static_cast<int>(laser_pos.y)),
                       4, cv::Scalar(0, 255, 255), 2);
            cv::line(output.canvas,
                     cv::Point(static_cast<int>(laser_pos.x), static_cast<int>(laser_pos.y)),
                     cv::Point(static_cast<int>(target_pos.x), static_cast<int>(target_pos.y)),
                     cv::Scalar(0, 255, 255),
                     1);
            std::ostringstream laser_line;
            laser_line << std::fixed << std::setprecision(2)
                       << "Laser->Target err(px): " << output.laser_target_error_px;
            cv::putText(output.canvas, laser_line.str(), {20, 210}, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(0, 255, 255), 2);
        }

        cv::drawMarker(output.canvas, cv::Point(static_cast<int>(cx), static_cast<int>(cy)),
                       cv::Scalar(0, 255, 0), cv::MARKER_CROSS, 14, 2);
    } else {
        output.canvas = frame;
    }

    return output;
}

float TargetTrackingPipeline::get_pitch_angle() const {
    return pitch_angle_;
}

float TargetTrackingPipeline::get_yaw_angle() const {
    return yaw_angle_;
}

bool TargetTrackingPipeline::open_serial() {
    if (!config_.enable_serial) {
        return false;
    }
    return gimbal_.open_serial(config_.serial_device, config_.serial_baud);
}

void TargetTrackingPipeline::close_serial() {
    gimbal_.close_serial();
}

bool TargetTrackingPipeline::is_serial_open() const {
    return gimbal_.is_serial_open();
}

bool TargetTrackingPipeline::send_raw_serial_command(const std::string& command) {
    return gimbal_.send_raw_command(command);
}

float TargetTrackingPipeline::clamp_value(float v, float lo, float hi) const {
    return std::max(lo, std::min(v, hi));
}

float TargetTrackingPipeline::pid_step(float error, float dt, PID& pid, float integral_limit, bool allow_integral) {
    if (allow_integral) {
        pid.integral += error * dt;
        pid.integral = clamp_value(pid.integral, -integral_limit, integral_limit);
    }

    float derivative = 0.0f;
    if (pid.has_prev && dt > 1e-6f) {
        derivative = (error - pid.prev_error) / dt;
    }
    pid.prev_error = error;
    pid.has_prev = true;

    return pid.kp * error + pid.ki * pid.integral + pid.kd * derivative;
}

void TargetTrackingPipeline::reset_pid(PID& pid) {
    pid.integral = 0.0f;
    pid.prev_error = 0.0f;
    pid.has_prev = false;
}

void TargetTrackingPipeline::apply_pid_gains_from_config() {
    pid_pitch_.kp = config_.pid_kp;
    pid_pitch_.ki = config_.pid_ki;
    pid_pitch_.kd = config_.pid_kd;
    pid_yaw_.kp = config_.pid_kp;
    pid_yaw_.ki = config_.pid_ki;
    pid_yaw_.kd = config_.pid_kd;
    reset_pid(pid_pitch_);
    reset_pid(pid_yaw_);
}

void TargetTrackingPipeline::reset_micro_sim_state() {
    sim_initialized_ = false;
    sim_time_ = 0.0f;
    sim_angle_rad_ = 0.0f;
    sim_speed_rad_s_ = 0.0f;
    sim_target_speed_rad_s_ = 0.0f;
    sim_no_measure_time_ = 0.0f;
    sim_orbit_center_px_ = cv::Point2f(-1.0f, -1.0f);
    sim_orbit_radius_px_ = 0.0f;
    sim_orbit_valid_ = false;

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> a_dist(config_.micro_sim_big_a_min, config_.micro_sim_big_a_max);
    std::uniform_real_distribution<float> w_dist(config_.micro_sim_big_omega_min, config_.micro_sim_big_omega_max);
    std::uniform_int_distribution<int> dir_dist(0, 1);

    sim_big_a_ = a_dist(rng);
    sim_big_omega_ = w_dist(rng);
    sim_big_b_ = 2.090f - sim_big_a_;
    sim_direction_sign_ = (config_.micro_sim_random_direction && dir_dist(rng) == 0) ? -1 : 1;

    std::ostringstream eq;
    eq << std::fixed << std::setprecision(3);
    if (config_.micro_sim_mode == MicroSimMode::SmallFixed) {
        eq << "w(t)=" << static_cast<float>(sim_direction_sign_) * config_.micro_sim_small_speed_rad_s << " rad/s";
    } else if (config_.micro_sim_mode == MicroSimMode::BigVariable) {
        eq << "w(t)=" << static_cast<float>(sim_direction_sign_) * sim_big_a_
           << "*sin(" << sim_big_omega_ << "*t)+"
           << static_cast<float>(sim_direction_sign_) * sim_big_b_;
    } else {
        eq << "sim disabled";
    }
    sim_equation_text_ = eq.str();
}

void TargetTrackingPipeline::update_micro_sim_dynamics(float dt) {
    if (!config_.enable_micro_sim || config_.micro_sim_mode == MicroSimMode::Disabled) {
        return;
    }

    sim_time_ += dt;
    float target_speed = 0.0f;
    if (config_.micro_sim_mode == MicroSimMode::SmallFixed) {
        target_speed = config_.micro_sim_small_speed_rad_s;
    } else {
        target_speed = sim_big_a_ * std::sin(sim_big_omega_ * sim_time_) + sim_big_b_;
    }
    target_speed *= static_cast<float>(sim_direction_sign_);
    sim_target_speed_rad_s_ = target_speed;

    const float tau = std::max(1e-3f, config_.micro_sim_speed_lag_sec);
    const float alpha = clamp_value(dt / tau, 0.0f, 1.0f);
    sim_speed_rad_s_ += (target_speed - sim_speed_rad_s_) * alpha;
    sim_angle_rad_ += sim_speed_rad_s_ * dt;
    if (sim_angle_rad_ > static_cast<float>(CV_PI)) {
        sim_angle_rad_ -= 2.0f * static_cast<float>(CV_PI);
    } else if (sim_angle_rad_ < -static_cast<float>(CV_PI)) {
        sim_angle_rad_ += 2.0f * static_cast<float>(CV_PI);
    }
}

void TargetTrackingPipeline::sync_micro_sim_orbit_from_measurement(const TargetInfo& info) {
    if (!info.found || info.board_center.x < 0.0f || info.target_center.x < 0.0f) {
        return;
    }

    const cv::Point2f center = info.board_center;
    const cv::Point2f target = info.target_center;
    const float radius = cv::norm(target - center);
    if (radius < 3.0f) {
        return;
    }

    const float measured_angle = std::atan2(target.y - center.y, target.x - center.x);
    if (!sim_orbit_valid_) {
        sim_orbit_center_px_ = center;
        sim_orbit_radius_px_ = radius;
        sim_angle_rad_ = measured_angle;
        sim_orbit_valid_ = true;
        sim_initialized_ = true;
        return;
    }

    const float kCenterBlend = 0.25f;
    const float kRadiusBlend = 0.20f;
    const float kAngleBlend = 0.25f;
    sim_orbit_center_px_ = sim_orbit_center_px_ * (1.0f - kCenterBlend) + center * kCenterBlend;
    sim_orbit_radius_px_ = sim_orbit_radius_px_ * (1.0f - kRadiusBlend) + radius * kRadiusBlend;

    float angle_diff = measured_angle - sim_angle_rad_;
    while (angle_diff > static_cast<float>(CV_PI)) angle_diff -= 2.0f * static_cast<float>(CV_PI);
    while (angle_diff < -static_cast<float>(CV_PI)) angle_diff += 2.0f * static_cast<float>(CV_PI);
    sim_angle_rad_ += kAngleBlend * angle_diff;
}

bool TargetTrackingPipeline::get_micro_sim_prediction(float dt, cv::Point2f& out_pos, bool account_missing) {
    if (!config_.enable_micro_sim || config_.micro_sim_mode == MicroSimMode::Disabled) {
        return false;
    }
    if (!sim_orbit_valid_ || !sim_initialized_) {
        return false;
    }

    if (account_missing) {
        sim_no_measure_time_ += dt;
    }
    if (sim_no_measure_time_ > std::max(0.05f, config_.micro_sim_prediction_hold_sec)) {
        return false;
    }

    const float lead_t = std::max(0.0f, dt);
    const float pred_angle = sim_angle_rad_ + sim_speed_rad_s_ * lead_t;

    out_pos = cv::Point2f(
        sim_orbit_center_px_.x + sim_orbit_radius_px_ * std::cos(pred_angle),
        sim_orbit_center_px_.y + sim_orbit_radius_px_ * std::sin(pred_angle));
    return true;
}
