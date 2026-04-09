#ifndef GIMBAL_CONTROL_HPP
#define GIMBAL_CONTROL_HPP
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <algorithm>
#include <cmath>

#include "TargetTracking/SerialPort.hpp"

class ServoMotor{
public:
    /// 创建舵机对象并记录ID。
    ServoMotor(int id) : _id(id),_angle(0.0f),_speed(0.0f){
        std::cout<<"ServoMotor "<<_id<<" created."<<std::endl;
    };
    /// 设置目标角度（单位：度）。
    void set_angle(float angle){
        _angle = angle;
    }
    /// 在当前角度基础上叠加增量。
    void add_angle(float delta){
        _angle += delta;
    }
    /// 获取当前目标角度。
    float get_angle() const{
        return _angle;
    }
    /// 设置角速度（单位：度/秒）。
    void set_speed(float speed){
        _speed = speed;
    }
    /// 获取当前角速度设定值。
    float get_speed() const{
        return _speed;
    }
    /// 根据目标角与速度更新 PWM/时间等内部状态。
    void prepare_motion(float min_angle_deg, float max_angle_deg){
        const float clamped = std::clamp(_angle, min_angle_deg, max_angle_deg);
        _pwm = angle_to_pwm(clamped, min_angle_deg, max_angle_deg);
        _time_ms = compute_time_ms(clamped);
        _last_angle = clamped;
        rebuild_command_buffer();
    }
    /// 根据零位角更新 PWM/时间等内部状态。
    void prepare_motion_with_zero(float min_angle_deg, float max_angle_deg, float zero_angle_deg){
        const float clamped = std::clamp(_angle, min_angle_deg, max_angle_deg);
        constexpr float kPwmPerDeg = 2000.0f / 270.0f;
        const float delta_deg = clamped - zero_angle_deg;
        const float pwm = 1500.0f + delta_deg * kPwmPerDeg;
        _pwm = std::clamp(static_cast<int>(std::lround(pwm)), 500, 2500);
        _time_ms = compute_time_ms(clamped);
        _last_angle = clamped;
        rebuild_command_buffer();
    }
    /// 兼容旧接口：按默认线性映射更新状态并刷新命令缓冲。
    void generate_command(float min_angle_deg, float max_angle_deg){
        prepare_motion(min_angle_deg, max_angle_deg);
    }
    /// 兼容旧接口：按零位映射更新状态并刷新命令缓冲。
    void generate_command(float min_angle_deg, float max_angle_deg, float zero_angle_deg){
        prepare_motion_with_zero(min_angle_deg, max_angle_deg, zero_angle_deg);
    }
    /// 返回最近一次生成的命令缓冲。
    const std::string& get_command_buffer() const{
        return _cmd;
    }
    /// 返回最近一次生成的PWM值。
    int get_pwm() const{
        return _pwm;
    }
    /// 返回最近一次生成的运动时长（ms）。
    int get_time_ms() const{
        return _time_ms;
    }
private:
    int _id;
    float _angle;
    float _speed;
    float _last_angle{0.0f};
    int _pwm{1500};
    int _time_ms{0};
    std::string _cmd;

    /// 将角度线性映射到 500-2500 PWM。
    static int angle_to_pwm(float angle_deg, float min_angle_deg, float max_angle_deg){
        const float span = max_angle_deg - min_angle_deg;
        if (span <= 0.0f) return 1500;
        const float t = (angle_deg - min_angle_deg) / span;
        const float pwm = 500.0f + t * (2500.0f - 500.0f);
        return static_cast<int>(std::lround(pwm));
    }
    /// 根据角度差和速度计算执行时间，并约束到协议范围。
    int compute_time_ms(float clamped_angle) const{
        if (_speed <= 0.0f) {
            return 0;
        }
        const int t = static_cast<int>(std::abs(clamped_angle - _last_angle) / _speed * 1000.0f);
        return std::clamp(t, 0, 9999);
    }
    /// 根据当前 PWM/时间状态重建单舵机命令字符串。
    void rebuild_command_buffer(){
        std::ostringstream oss;
        oss << "#" << std::setw(3) << std::setfill('0') << _id
            << "P" << std::setw(4) << std::setfill('0') << _pwm
            << "T" << std::setw(4) << std::setfill('0') << _time_ms << "!";
        _cmd = oss.str();
    }
};

class GimbalControl{
public:
    /// 初始化双轴云台控制器。
    GimbalControl(){
        std::cout << "Gimbal created." << std::endl;
    }
    /// 设置俯仰轴PWM零位对应角度。
    void set_pitch_zero_angle_deg(float angle){
        _pitch_zero_angle_deg = angle;
    }
    /// 设置偏航轴PWM零位对应角度。
    void set_yaw_zero_angle_deg(float angle){
        _yaw_zero_angle_deg = angle;
    }
    /// 设置俯仰目标角度。
    void set_pitch_angle(float angle){
        _pitch_motor.set_angle(angle);
    }
    /// 设置偏航目标角度。
    void set_yaw_angle(float angle){
        _yaw_motor.set_angle(angle);
    }
    /// 设置俯仰速度约束。
    void set_pitch_speed(float speed){
        _pitch_motor.set_speed(speed);
    }
    /// 设置偏航速度约束。
    void set_yaw_speed(float speed){
        _yaw_motor.set_speed(speed);
    }
    /// 组合各通道并生成整包云台控制命令。
    void get_command(){
        _yaw_motor.prepare_motion_with_zero(0.0f, 270.0f, _yaw_zero_angle_deg);
        _pitch_motor.prepare_motion_with_zero(0.0f, 270.0f, _pitch_zero_angle_deg);

        const int yaw_pwm = _yaw_motor.get_pwm();
        const int pitch_pwm = _pitch_motor.get_pwm();
        const int yaw_t = _yaw_motor.get_time_ms();
        const int pitch_t = _pitch_motor.get_time_ms();

        std::ostringstream oss;
        oss << "{"
            << "P" << std::setw(4) << std::setfill('0') << yaw_pwm
            << "T" << std::setw(4) << std::setfill('0') << yaw_t
            << "P1350T1000"
            << "P2300T1000"
            << "P" << std::setw(4) << std::setfill('0') << pitch_pwm
            << "T" << std::setw(4) << std::setfill('0') << pitch_t
            << "P1500T1000"
            << "}";
        _command_buffer = oss.str();
        std::cout << "Generated command: " << _command_buffer << std::endl;
    }
    /// 获取当前完整命令字符串。
    const std::string& get_command_buffer() const{
        return _command_buffer;
    }
    /// 打开串口设备用于实际发送控制命令。
    bool open_serial(const std::string& device, int baudrate){
        return _serial.open(device, baudrate);
    }
    /// 关闭串口。
    void close_serial(){
        _serial.close();
    }
    /// 查询串口是否可用。
    bool is_serial_open() const{
        return _serial.is_open();
    }
    /// 发送当前自动生成命令。
    bool send_command(){
        if (!_serial.is_open()) {
            std::cout << "Sending command (serial closed): " << get_command_buffer() << std::endl;
            return false;
        }
        return _serial.write_string(get_command_buffer());
    }
    /// 发送用户提供的原始命令。
    bool send_raw_command(const std::string& command){
        if (!_serial.is_open()) {
            std::cout << "Sending raw command (serial closed): " << command << std::endl;
            return false;
        }
        return _serial.write_string(command);
    }

private:
    ServoMotor _pitch_motor{3};
    ServoMotor _yaw_motor{0};
    float _pitch_zero_angle_deg{135.0f};
    float _yaw_zero_angle_deg{135.0f};
    std::string _command_buffer;
    SerialPort _serial;
};

#endif // GIMBAL_CONTROL_HPP