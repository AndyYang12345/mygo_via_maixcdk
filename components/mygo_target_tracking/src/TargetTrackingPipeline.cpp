#include "TargetTracking/TargetTrackingPipeline.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace {
float wrap_angle_rad(float angle_rad) {
    while (angle_rad > static_cast<float>(CV_PI)) {
        angle_rad -= 2.0f * static_cast<float>(CV_PI);
    }
    while (angle_rad < -static_cast<float>(CV_PI)) {
        angle_rad += 2.0f * static_cast<float>(CV_PI);
    }
    return angle_rad;
}
}

TargetTrackingPipeline::TargetTrackingPipeline() {
    control_enabled_ = config_.control_enabled;
    gimbal_.set_pitch_zero_angle_deg(config_.pitch_pwm_zero_angle);
    gimbal_.set_yaw_zero_angle_deg(config_.yaw_pwm_zero_angle);
    apply_pid_gains_from_config();
    pitch_angle_ = config_.pitch_home;
    yaw_angle_ = config_.yaw_home;
}

void TargetTrackingPipeline::set_config(const PipelineConfig& config) {
    config_ = config;
    control_enabled_ = config_.control_enabled;
    gimbal_.set_pitch_zero_angle_deg(config_.pitch_pwm_zero_angle);
    gimbal_.set_yaw_zero_angle_deg(config_.yaw_pwm_zero_angle);
    apply_pid_gains_from_config();
    pitch_angle_ = config_.pitch_home;
    yaw_angle_ = config_.yaw_home;
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
    reset_pid(pid_pitch_);
    reset_pid(pid_yaw_);

    if (config_.enable_open_loop_phase_orbit) {
        open_loop_phase_active_ = true;
        open_loop_phase_rad_ = config_.open_loop_phase_init_rad;
        open_loop_base_pitch_deg_ = pitch_angle_;
        open_loop_base_yaw_deg_ = yaw_angle_;
        open_loop_distance_mm_ = std::max(1.0f, config_.open_loop_default_distance_mm);
        open_loop_locked_omega_rad_s_ = config_.open_loop_omega_rad_s;
        open_loop_locked_from_fit_ = false;
        phase_corr_integral_ = 0.0f;
        phase_corr_omega_bias_rad_s_ = 0.0f;
        phase_corr_outlier_count_ = 0;
        phase_corr_freeze_left_ = 0;

        if (speed_id_valid_ &&
            speed_id_omega_count_ >= std::max(1, config_.speed_id_min_samples) &&
            std::isfinite(speed_id_omega_rad_s_) &&
            std::abs(speed_id_omega_rad_s_) > 1e-4f) {
            open_loop_locked_omega_rad_s_ = speed_id_omega_rad_s_;
            open_loop_locked_from_fit_ = true;
            open_loop_phase_rad_ = wrap_angle_rad(speed_id_last_phase_rad_);
        }
    } else {
        open_loop_phase_active_ = false;
    }
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
    open_loop_phase_active_ = false;
    open_loop_phase_rad_ = 0.0f;
    open_loop_base_pitch_deg_ = config_.pitch_home;
    open_loop_base_yaw_deg_ = config_.yaw_home;
    open_loop_distance_mm_ = -1.0f;
    open_loop_locked_omega_rad_s_ = config_.open_loop_omega_rad_s;
    open_loop_locked_from_fit_ = false;
    phase_corr_integral_ = 0.0f;
    phase_corr_omega_bias_rad_s_ = 0.0f;
    phase_corr_outlier_count_ = 0;
    phase_corr_freeze_left_ = 0;
    last_board_distance_mm_ = -1.0f;
    speed_id_active_ = false;
    speed_id_valid_ = false;
    speed_id_event_pending_ = false;
    speed_id_samples_ = 0;
    speed_id_all_in_roi_ = true;
    speed_id_warmup_elapsed_s_ = 0.0f;
    speed_id_validate_elapsed_s_ = 0.0f;
    speed_id_phase_init_rad_ = 0.0f;
    speed_id_last_phase_rad_ = 0.0f;
    speed_id_unwrapped_phase_rad_ = 0.0f;
    speed_id_unwrapped_phase_start_rad_ = 0.0f;
    speed_id_omega_rad_s_ = config_.open_loop_omega_rad_s;
    speed_id_inst_omega_rad_s_ = 0.0f;
    speed_id_radius_px_ = 0.0f;
    speed_id_validation_started_ = false;
    speed_id_has_prev_vec_ = false;
    speed_id_prev_vec_ = cv::Point2f(0.0f, 0.0f);
    speed_id_omega_sum_rad_s_ = 0.0f;
    speed_id_omega_count_ = 0;
    speed_id_reg_sum_t_ = 0.0f;
    speed_id_reg_sum_p_ = 0.0f;
    speed_id_reg_sum_tt_ = 0.0f;
    speed_id_reg_sum_tp_ = 0.0f;
    speed_id_last_error_px_ = -1.0f;
    speed_id_last_tolerance_px_ = -1.0f;
    speed_id_last_predicted_pos_ = cv::Point2f(-1.0f, -1.0f);
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

std::pair<float, float> TargetTrackingPipeline::compute_servo_angles_from_offsets(
    float current_pitch_deg,
    float current_yaw_deg,
    float yaw_offset_rad,
    float pitch_offset_rad) const {
    const float rad_to_deg = 180.0f / static_cast<float>(CV_PI);
    const float next_yaw_deg = clamp_value(
        current_yaw_deg + yaw_offset_rad * rad_to_deg,
        0.0f,
        270.0f);
    const float next_pitch_deg = clamp_value(
        current_pitch_deg + pitch_offset_rad * rad_to_deg,
        0.0f,
        270.0f);
    return {next_pitch_deg, next_yaw_deg};
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
    output.view_angle_valid = info.view_angle_valid;
    output.view_delta_yaw_rad = 0.0f;
    output.view_delta_pitch_rad = 0.0f;
    output.feedforward_pitch_angle = pitch_angle_;
    output.feedforward_yaw_angle = yaw_angle_;
    output.open_loop_active = false;
    output.open_loop_phase_rad = open_loop_phase_rad_;
    output.open_loop_omega_rad_s = open_loop_locked_omega_rad_s_;
    output.open_loop_distance_mm = open_loop_distance_mm_;
    output.phase_correction_active = false;
    output.phase_error_rad = 0.0f;
    output.phase_correction_step_rad = 0.0f;
    output.phase_correction_omega_bias_rad_s = phase_corr_omega_bias_rad_s_;
    output.phase_correction_skipped = false;
    output.phase_correction_outlier_count = phase_corr_outlier_count_;
    output.predicted_pos_valid = false;
    output.predicted_pos = cv::Point2f(-1.0f, -1.0f);
    output.speed_identifying = speed_id_active_;
    output.speed_identified = speed_id_valid_;
    output.speed_identified_event = false;
    output.identified_omega_rad_s = speed_id_valid_ ? speed_id_omega_rad_s_ : config_.open_loop_omega_rad_s;
    output.identified_phase_rad = wrap_angle_rad(speed_id_last_phase_rad_);
    output.instant_omega_rad_s = speed_id_inst_omega_rad_s_;
    output.fitted_omega_rad_s = speed_id_omega_rad_s_;
    output.speed_fit_samples = speed_id_omega_count_;
    output.speed_validation_error_px = speed_id_last_error_px_;
    output.speed_validation_tolerance_px = speed_id_last_tolerance_px_;

    if (info.board_distance_mm > 0.0f) {
        last_board_distance_mm_ = info.board_distance_mm;
    }

    if (info.view_angle_valid) {
        output.view_delta_yaw_rad = info.view_delta_x[0];
        output.view_delta_pitch_rad = info.view_delta_y[0];
        const auto feedforward_angles = compute_servo_angles_from_offsets(
            pitch_angle_,
            yaw_angle_,
            output.view_delta_yaw_rad,
            output.view_delta_pitch_rad);
        output.feedforward_pitch_angle = feedforward_angles.first;
        output.feedforward_yaw_angle = feedforward_angles.second;
    }

    if (state_ == TrackState::Waiting) {
        pitch_angle_ = config_.pitch_home;
        yaw_angle_ = config_.yaw_home;
        pitch_speed_ = 0.0f;
        yaw_speed_ = 0.0f;
    } else if (state_ == TrackState::Searching) {
        if (config_.enable_speed_identification && has_target && info.roi_active) {
            const cv::Point2f delta = info.target_center - info.board_center;
            const float phase_rad = std::atan2(delta.y, delta.x);
            const float radius_px = std::max(1.0f, static_cast<float>(cv::norm(delta)));

            if (!speed_id_active_) {
                speed_id_active_ = true;
                speed_id_valid_ = false;
                speed_id_samples_ = 0;
                speed_id_all_in_roi_ = true;
                speed_id_warmup_elapsed_s_ = 0.0f;
                speed_id_validate_elapsed_s_ = 0.0f;
                speed_id_phase_init_rad_ = phase_rad;
                speed_id_last_phase_rad_ = phase_rad;
                speed_id_unwrapped_phase_rad_ = phase_rad;
                speed_id_unwrapped_phase_start_rad_ = phase_rad;
                speed_id_omega_rad_s_ = config_.open_loop_omega_rad_s;
                speed_id_inst_omega_rad_s_ = 0.0f;
                speed_id_radius_px_ = radius_px;
                speed_id_validation_started_ = false;
                speed_id_has_prev_vec_ = true;
                speed_id_prev_vec_ = delta;
                speed_id_omega_sum_rad_s_ = 0.0f;
                speed_id_omega_count_ = 0;
                speed_id_reg_sum_t_ = 0.0f;
                speed_id_reg_sum_p_ = 0.0f;
                speed_id_reg_sum_tt_ = 0.0f;
                speed_id_reg_sum_tp_ = 0.0f;
                speed_id_last_error_px_ = -1.0f;
                speed_id_last_tolerance_px_ = -1.0f;
                speed_id_last_predicted_pos_ = info.target_center;

                if (config_.print_debug) {
                    std::cout << "[SpeedID] start from ROI target" << std::endl;
                }
            } else {
                speed_id_radius_px_ = (1.0f - config_.speed_id_radius_smooth_alpha) * speed_id_radius_px_ +
                                      config_.speed_id_radius_smooth_alpha * radius_px;

                float inst_omega = 0.0f;
                bool inst_omega_valid = false;
                if (speed_id_has_prev_vec_) {
                    const float cross = speed_id_prev_vec_.x * delta.y - speed_id_prev_vec_.y * delta.x;
                    const float dot = speed_id_prev_vec_.x * delta.x + speed_id_prev_vec_.y * delta.y;
                    const float dphase = std::atan2(cross, dot);
                    speed_id_unwrapped_phase_rad_ += dphase;
                    inst_omega = dphase / std::max(1e-4f, dt);
                    inst_omega_valid = std::isfinite(inst_omega);
                }
                speed_id_prev_vec_ = delta;
                speed_id_has_prev_vec_ = true;
                speed_id_last_phase_rad_ = phase_rad;
                if (inst_omega_valid) {
                    speed_id_inst_omega_rad_s_ = inst_omega;
                }

                speed_id_warmup_elapsed_s_ += dt;
                if (speed_id_warmup_elapsed_s_ >= config_.speed_id_warmup_s) {
                    if (!speed_id_validation_started_) {
                        speed_id_validation_started_ = true;
                        speed_id_phase_init_rad_ = phase_rad;
                        speed_id_unwrapped_phase_start_rad_ = speed_id_unwrapped_phase_rad_;
                        speed_id_validate_elapsed_s_ = 0.0f;
                        speed_id_samples_ = 0;
                        speed_id_omega_sum_rad_s_ = 0.0f;
                        speed_id_omega_count_ = 0;
                        speed_id_reg_sum_t_ = 0.0f;
                        speed_id_reg_sum_p_ = 0.0f;
                        speed_id_reg_sum_tt_ = 0.0f;
                        speed_id_reg_sum_tp_ = 0.0f;
                    } else {
                        speed_id_validate_elapsed_s_ += dt;
                        speed_id_samples_++;

                        if (inst_omega_valid) {
                            speed_id_omega_sum_rad_s_ += inst_omega;
                            speed_id_omega_count_++;
                        }

                        const float t = speed_id_validate_elapsed_s_;
                        const float p = speed_id_unwrapped_phase_rad_ - speed_id_unwrapped_phase_start_rad_;
                        speed_id_reg_sum_t_ += t;
                        speed_id_reg_sum_p_ += p;
                        speed_id_reg_sum_tt_ += t * t;
                        speed_id_reg_sum_tp_ += t * p;

                        if (speed_id_samples_ >= 2) {
                            const float n = static_cast<float>(speed_id_samples_);
                            const float denom = n * speed_id_reg_sum_tt_ - speed_id_reg_sum_t_ * speed_id_reg_sum_t_;
                            if (std::abs(denom) > 1e-6f) {
                                speed_id_omega_rad_s_ =
                                    (n * speed_id_reg_sum_tp_ - speed_id_reg_sum_t_ * speed_id_reg_sum_p_) / denom;
                            } else if (speed_id_omega_count_ > 0) {
                                speed_id_omega_rad_s_ = speed_id_omega_sum_rad_s_ /
                                                    static_cast<float>(speed_id_omega_count_);
                            }
                        } else if (speed_id_omega_count_ > 0) {
                            speed_id_omega_rad_s_ = speed_id_omega_sum_rad_s_ /
                                                    static_cast<float>(speed_id_omega_count_);
                        }

                        const float predicted_phase = speed_id_phase_init_rad_ +
                                                      speed_id_omega_rad_s_ * speed_id_validate_elapsed_s_;
                        const cv::Point2f predicted(
                            info.board_center.x + speed_id_radius_px_ * std::cos(predicted_phase),
                            info.board_center.y + speed_id_radius_px_ * std::sin(predicted_phase));
                        const float error_px = cv::norm(predicted - info.target_center);
                        const float roi_span = static_cast<float>(std::max(1, std::min(info.roi_rect.width, info.roi_rect.height)));
                        const float tolerance_px = std::max(config_.speed_id_min_tolerance_px,
                                                            config_.speed_id_roi_tolerance_ratio * roi_span);

                        speed_id_last_predicted_pos_ = predicted;
                        speed_id_last_error_px_ = error_px;
                        speed_id_last_tolerance_px_ = tolerance_px;

                        if (error_px > tolerance_px) {
                            speed_id_all_in_roi_ = false;
                        }

                        output.predicted_pos_valid = true;
                        output.predicted_pos = predicted;
                        output.speed_validation_error_px = error_px;
                        output.speed_validation_tolerance_px = tolerance_px;
                        output.identified_omega_rad_s = speed_id_omega_rad_s_;
                        output.identified_phase_rad = wrap_angle_rad(speed_id_last_phase_rad_);
                        output.instant_omega_rad_s = speed_id_inst_omega_rad_s_;
                        output.fitted_omega_rad_s = speed_id_omega_rad_s_;
                        output.speed_fit_samples = speed_id_samples_;
                    }
                }

                if (speed_id_validate_elapsed_s_ >= config_.speed_id_validate_s) {
                    if (speed_id_all_in_roi_ && speed_id_samples_ >= config_.speed_id_min_samples) {
                        speed_id_valid_ = true;
                        speed_id_event_pending_ = true;
                        state_ = TrackState::Locked;
                        lock_count_ = 0;
                        lost_count_ = 0;
                        pitch_speed_ = 0.0f;
                        yaw_speed_ = 0.0f;

                        if (config_.print_debug) {
                            std::cout << std::fixed << std::setprecision(4)
                                      << "[SpeedID] success omega=" << speed_id_omega_rad_s_
                                      << " phase=" << wrap_angle_rad(speed_id_last_phase_rad_)
                                      << " rad" << std::endl;
                        }
                    } else {
                        speed_id_active_ = false;
                        speed_id_samples_ = 0;
                        speed_id_warmup_elapsed_s_ = 0.0f;
                        speed_id_validate_elapsed_s_ = 0.0f;
                        speed_id_all_in_roi_ = true;
                        speed_id_validation_started_ = false;
                        speed_id_has_prev_vec_ = false;
                        speed_id_prev_vec_ = cv::Point2f(0.0f, 0.0f);
                        speed_id_omega_sum_rad_s_ = 0.0f;
                        speed_id_omega_count_ = 0;
                        speed_id_reg_sum_t_ = 0.0f;
                        speed_id_reg_sum_p_ = 0.0f;
                        speed_id_reg_sum_tt_ = 0.0f;
                        speed_id_reg_sum_tp_ = 0.0f;

                        if (config_.print_debug) {
                            std::cout << "[SpeedID] retry due to prediction mismatch" << std::endl;
                        }
                    }
                }
            }

            pitch_speed_ = 0.0f;
            yaw_speed_ = 0.0f;
            output.speed_identifying = speed_id_active_;
            output.speed_identified = speed_id_valid_;
        } else {
            speed_id_active_ = false;
            speed_id_samples_ = 0;
            speed_id_warmup_elapsed_s_ = 0.0f;
            speed_id_validate_elapsed_s_ = 0.0f;
            speed_id_all_in_roi_ = true;
            speed_id_validation_started_ = false;
            speed_id_has_prev_vec_ = false;
            speed_id_prev_vec_ = cv::Point2f(0.0f, 0.0f);
            speed_id_omega_sum_rad_s_ = 0.0f;
            speed_id_omega_count_ = 0;
            speed_id_reg_sum_t_ = 0.0f;
            speed_id_reg_sum_p_ = 0.0f;
            speed_id_reg_sum_tt_ = 0.0f;
            speed_id_reg_sum_tp_ = 0.0f;

        if (config_.enable_view_angle_feedforward && has_target && info.view_angle_valid) {
            pitch_angle_ = output.feedforward_pitch_angle;
            yaw_angle_ = output.feedforward_yaw_angle;
            pitch_speed_ = 0.0f;
            yaw_speed_ = 0.0f;

            if (config_.print_debug) {
                std::cout << std::fixed << std::setprecision(2)
                          << "[FF] board_distance_mm:" << output.board_distance_mm
                          << " yaw_offset_rad:" << output.view_delta_yaw_rad
                          << " pitch_offset_rad:" << output.view_delta_pitch_rad
                          << " next_pitch_deg:" << output.feedforward_pitch_angle
                          << " next_yaw_deg:" << output.feedforward_yaw_angle
                          << std::endl;
            }
        } else {
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
        }
        }
    } else if (state_ == TrackState::Locked) {
        pitch_speed_ = 0.0f;
        yaw_speed_ = 0.0f;
        output.speed_identifying = false;
        output.speed_identified = speed_id_valid_;
        output.identified_omega_rad_s = speed_id_valid_ ? speed_id_omega_rad_s_ : open_loop_locked_omega_rad_s_;
        if (speed_id_last_predicted_pos_.x >= 0.0f && speed_id_last_predicted_pos_.y >= 0.0f) {
            output.predicted_pos_valid = true;
            output.predicted_pos = speed_id_last_predicted_pos_;
        }
        if (has_target) {
            lost_count_ = 0;
        } else {
            lost_count_++;
            if (lost_count_ >= config_.lost_required) {
                output.tracking_recovery_requested = true;
                reset();
            }
        }
    } else if (state_ == TrackState::Tracking) {
        if (config_.enable_open_loop_phase_orbit && open_loop_phase_active_) {
            const TrackerConfig tracker_cfg = tracker_.get_config();
            const float orbit_radius_mm = std::max(1.0f, tracker_cfg.target_orbit_radius_mm);
            const float orbit_omega_rad_s = open_loop_locked_omega_rad_s_ + phase_corr_omega_bias_rad_s_;
            const float board_distance_mm = std::max(
                1.0f,
                (open_loop_distance_mm_ > 0.0f)
                    ? open_loop_distance_mm_
                    : std::max(1.0f, config_.open_loop_default_distance_mm));

            const float prev_pitch = pitch_angle_;
            const float prev_yaw = yaw_angle_;

            open_loop_phase_rad_ += orbit_omega_rad_s * dt;

            if (config_.enable_phase_correction && has_target && info.roi_active) {
                const cv::Point2f meas_delta = info.target_center - info.board_center;
                const float meas_phase_rad = std::atan2(meas_delta.y, meas_delta.x);
                const float phase_error = wrap_angle_rad(meas_phase_rad - open_loop_phase_rad_);

                const bool innovation_outlier =
                    std::abs(phase_error) > std::max(0.01f, config_.phase_corr_innovation_gate_rad);
                if (innovation_outlier) {
                    phase_corr_outlier_count_++;
                    if (phase_corr_freeze_left_ <= 0) {
                        phase_corr_freeze_left_ = std::max(0, config_.phase_corr_outlier_freeze_frames);
                    }
                } else {
                    phase_corr_outlier_count_ = 0;
                }

                float corr_step = 0.0f;
                bool skipped = false;
                if (phase_corr_freeze_left_ > 0) {
                    skipped = true;
                    phase_corr_freeze_left_--;
                } else {
                    phase_corr_integral_ += phase_error * dt;
                    phase_corr_integral_ = clamp_value(
                        phase_corr_integral_,
                        -config_.phase_corr_integral_limit,
                        config_.phase_corr_integral_limit);

                    phase_corr_omega_bias_rad_s_ += config_.phase_corr_ki * phase_error * dt;
                    phase_corr_omega_bias_rad_s_ = clamp_value(
                        phase_corr_omega_bias_rad_s_,
                        -config_.phase_corr_omega_bias_limit_rad_s,
                        config_.phase_corr_omega_bias_limit_rad_s);

                    corr_step = clamp_value(
                        config_.phase_corr_kp * phase_error,
                        -config_.phase_corr_max_step_rad,
                        config_.phase_corr_max_step_rad);
                    open_loop_phase_rad_ += corr_step;
                }

                output.phase_correction_active = true;
                output.phase_error_rad = phase_error;
                output.phase_correction_step_rad = corr_step;
                output.phase_correction_omega_bias_rad_s = phase_corr_omega_bias_rad_s_;
                output.phase_correction_skipped = skipped;
                output.phase_correction_outlier_count = phase_corr_outlier_count_;
            }

            const float x_mm = orbit_radius_mm * std::cos(open_loop_phase_rad_);
            const float y_mm = orbit_radius_mm * std::sin(open_loop_phase_rad_);
            const float yaw_offset_rad = std::atan2(-x_mm, board_distance_mm);
            const float pitch_offset_rad = std::atan2(-y_mm, board_distance_mm);

            const auto orbit_angles = compute_servo_angles_from_offsets(
                open_loop_base_pitch_deg_,
                open_loop_base_yaw_deg_,
                yaw_offset_rad,
                pitch_offset_rad);

            pitch_angle_ = orbit_angles.first;
            yaw_angle_ = orbit_angles.second;
            pitch_speed_ = (pitch_angle_ - prev_pitch) / dt;
            yaw_speed_ = (yaw_angle_ - prev_yaw) / dt;

            output.open_loop_active = true;
            output.open_loop_phase_rad = open_loop_phase_rad_;
            output.open_loop_omega_rad_s = orbit_omega_rad_s;
            output.open_loop_distance_mm = board_distance_mm;
            output.view_angle_valid = true;
            output.view_delta_yaw_rad = yaw_offset_rad;
            output.view_delta_pitch_rad = pitch_offset_rad;
            output.feedforward_pitch_angle = pitch_angle_;
            output.feedforward_yaw_angle = yaw_angle_;
            output.speed_identified = open_loop_locked_from_fit_;
            output.identified_omega_rad_s = orbit_omega_rad_s;

            lost_count_ = 0;
            reset_pid(pid_pitch_);
            reset_pid(pid_yaw_);
        } else if (!control_enabled_) {
            pitch_speed_ = 0.0f;
            yaw_speed_ = 0.0f;
        } else if (has_target) {
            float dx = 0.0f;
            float dy = 0.0f;
            dx = target_pos.x - cx;
            dy = target_pos.y - cy;
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
                          << "Target(px):(" << target_pos.x << "," << target_pos.y << ")"
                          << " Laser(px):(" << laser_pos.x << "," << laser_pos.y << ")"
                          << " dx,dy:(" << dx << "," << dy << ")"
                          << " pitch_err:" << pitch_error
                          << " yaw_err:" << yaw_error
                          << " cur_pitch:" << pitch_angle_
                          << " cur_yaw:" << yaw_angle_
                          << std::endl;
            }

            float pitch_speed_cmd = pid_step(pitch_error, dt, pid_pitch_, config_.integral_limit);
            float yaw_speed_cmd = pid_step(yaw_error, dt, pid_yaw_, config_.integral_limit);

            pitch_speed_ = clamp_value(pitch_speed_cmd, -config_.max_speed, config_.max_speed);
            yaw_speed_ = clamp_value(yaw_speed_cmd, -config_.max_speed, config_.max_speed);

            pitch_angle_ += pitch_speed_ * dt;
            yaw_angle_ += yaw_speed_ * dt;
            pitch_angle_ = clamp_value(pitch_angle_, 0.0f, 270.0f);
            yaw_angle_ = clamp_value(yaw_angle_, 0.0f, 270.0f);

            lost_count_ = 0;
        } else {
            pitch_speed_ = 0.0f;
            yaw_speed_ = 0.0f;
            lost_count_++;
            if (lost_count_ >= config_.lost_required) {
                output.tracking_recovery_requested = true;
                reset();
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
    output.target_pos = target_pos;
    output.board_pos = info.board_center;
    output.roi_active = info.roi_active;
    output.roi_rect = info.roi_rect;
    output.laser_found = has_laser;
    output.laser_pos = laser_pos;
    output.laser_target_error_px = (has_target && has_laser) ? cv::norm(target_pos - laser_pos) : 0.0f;
    output.board_distance_mm = info.board_distance_mm;
    if (speed_id_event_pending_) {
        output.speed_identified_event = true;
        speed_id_event_pending_ = false;
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
                mode_line << "Waiting | press SPACE to start scanning";
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
        if (output.predicted_pos_valid && output.predicted_pos.x >= 0.0f && output.predicted_pos.y >= 0.0f) {
            cv::drawMarker(output.canvas,
                           cv::Point(static_cast<int>(output.predicted_pos.x), static_cast<int>(output.predicted_pos.y)),
                           cv::Scalar(255, 80, 255),
                           cv::MARKER_TILTED_CROSS,
                           14,
                           2);
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

float TargetTrackingPipeline::pid_step(float error, float dt, PID& pid, float integral_limit) {
    pid.integral += error * dt;
    pid.integral = clamp_value(pid.integral, -integral_limit, integral_limit);

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
