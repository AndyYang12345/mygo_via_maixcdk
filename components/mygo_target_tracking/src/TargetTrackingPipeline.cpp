#include "TargetTracking/TargetTrackingPipeline.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>

TargetTrackingPipeline::TargetTrackingPipeline() {
    pitch_angle_ = clamp_value(config_.pitch_home, config_.pitch_min, config_.pitch_max);
    yaw_angle_ = clamp_value(config_.yaw_home, config_.yaw_min, config_.yaw_max);
}

void TargetTrackingPipeline::set_config(const PipelineConfig& config) {
    config_ = config;
    pitch_angle_ = clamp_value(config_.pitch_home, config_.pitch_min, config_.pitch_max);
    yaw_angle_ = clamp_value(config_.yaw_home, config_.yaw_min, config_.yaw_max);
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

void TargetTrackingPipeline::reset() {
    state_ = TrackState::Searching;
    lock_count_ = 0;
    lost_count_ = 0;
    scan_time_ = 0.0f;
    pitch_angle_ = clamp_value(config_.pitch_home, config_.pitch_min, config_.pitch_max);
    yaw_angle_ = clamp_value(config_.yaw_home, config_.yaw_min, config_.yaw_max);
    pitch_speed_ = 0.0f;
    yaw_speed_ = 0.0f;
    reset_pid(pid_pitch_);
    reset_pid(pid_yaw_);
    tracker_.reset_roi_tracking();
}

void TargetTrackingPipeline::handle_key(int key) {
    (void)key;
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
    cv::Point2f laser_pos(-1.0f, -1.0f);
    if (has_laser) {
        laser_pos = info.laser_center;
    }

    if (state_ == TrackState::Searching) {
        scan_time_ += dt;
        float yaw_phase = 2.0f * static_cast<float>(CV_PI) * config_.scan_yaw_freq * scan_time_;
        float pitch_phase = 2.0f * static_cast<float>(CV_PI) * config_.scan_pitch_freq * scan_time_ + config_.scan_phase;
        yaw_angle_ = config_.yaw_home + config_.scan_yaw_amp * std::sin(yaw_phase);
        pitch_angle_ = config_.pitch_home + config_.scan_pitch_amp * std::sin(pitch_phase);
        yaw_angle_ = clamp_value(yaw_angle_, config_.yaw_min, config_.yaw_max);
        pitch_angle_ = clamp_value(pitch_angle_, config_.pitch_min, config_.pitch_max);
        yaw_speed_ = config_.scan_yaw_amp * 2.0f * static_cast<float>(CV_PI) * config_.scan_yaw_freq * std::cos(yaw_phase);
        pitch_speed_ = config_.scan_pitch_amp * 2.0f * static_cast<float>(CV_PI) * config_.scan_pitch_freq * std::cos(pitch_phase);

        if (has_target) {
            lock_count_++;
        } else {
            lock_count_ = 0;
        }

        if (lock_count_ >= config_.lock_required) {
            state_ = TrackState::Tracking;
            reset_pid(pid_pitch_);
            reset_pid(pid_yaw_);
            lock_count_ = 0;
            lost_count_ = 0;
        }
    } else if (state_ == TrackState::Tracking) {
        if (has_target) {
            float dx = target_pos.x - cx;
            float dy = target_pos.y - cy;
            float pitch_error = std::atan2(dy, fy) * 180.0f / static_cast<float>(CV_PI);
            float yaw_error = -std::atan2(dx, fx) * 180.0f / static_cast<float>(CV_PI);

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
            pitch_angle_ = clamp_value(pitch_angle_, config_.pitch_min, config_.pitch_max);
            yaw_angle_ = clamp_value(yaw_angle_, config_.yaw_min, config_.yaw_max);

            lost_count_ = 0;
        } else {
            pitch_speed_ = 0.0f;
            yaw_speed_ = 0.0f;
            lost_count_++;
            if (lost_count_ >= config_.lost_required) {
                state_ = TrackState::Searching;
                lost_count_ = 0;
                lock_count_ = 0;
                scan_time_ = 0.0f;
            }
        }
    }

    gimbal_.set_pitch_angle(pitch_angle_);
    gimbal_.set_yaw_angle(yaw_angle_);
    gimbal_.set_pitch_speed(std::abs(pitch_speed_));
    gimbal_.set_yaw_speed(std::abs(yaw_speed_));
    gimbal_.get_command();

    const std::string cmd = gimbal_.get_command_buffer();
    if (config_.enable_serial && gimbal_.is_serial_open()) {
        gimbal_.send_command();
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
    output.laser_found = has_laser;
    output.laser_pos = laser_pos;
    output.laser_target_error_px = (has_target && has_laser) ? cv::norm(target_pos - laser_pos) : 0.0f;

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
            case TrackState::Searching:
                mode_line << "Searching(sin) | lock:" << lock_count_ << "/" << config_.lock_required;
                break;
            case TrackState::Tracking:
                mode_line << "Tracking(laser-hit) | lost:" << lost_count_ << "/" << config_.lost_required;
                break;
            case TrackState::Waiting:
            case TrackState::Locked:
                mode_line << "Searching(sin)";
                break;
        }
        mode_line << " | q/ESC=quit";
        cv::putText(output.canvas, mode_line.str(), {20, 430}, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(180, 180, 180), 1);

        if (has_target && target_pos.x >= 0.0f) {
            cv::circle(output.canvas, cv::Point(static_cast<int>(target_pos.x), static_cast<int>(target_pos.y)),
                       6, cv::Scalar(0, 0, 255), -1);
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
