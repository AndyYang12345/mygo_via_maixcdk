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
    ServoMotor(int id,
               float initial_angle_deg = 0.0f,
               float initial_min_angle_deg = 0.0f,
               float initial_max_angle_deg = 270.0f)
        : _id(id),_angle(initial_angle_deg),_speed(0.0f),_last_angle(initial_angle_deg){
        _pwm = angle_to_pwm(initial_angle_deg, initial_min_angle_deg, initial_max_angle_deg);
        std::cout<<"ServoMotor "<<_id<<" created."<<std::endl;
    };
    void set_angle(float angle){
        _angle = angle;
    }
    void add_angle(float delta){
        _angle += delta;
    }
    float get_angle() const{
        return _angle;
    }
    void set_speed(float speed){
        _speed = speed;
    }
    float get_speed() const{
        return _speed;
    }
    void generate_command(float min_angle_deg, float max_angle_deg){
        const float clamped = std::clamp(_angle, min_angle_deg, max_angle_deg);
        _pwm = angle_to_pwm(clamped, min_angle_deg, max_angle_deg, _min_pwm, _max_pwm);
        if (_speed > 0.0f) {
            _time_ms = static_cast<int>(std::abs(clamped - _last_angle) / _speed * 1000.0f);
        } else {
            _time_ms = 0;
        }
        _last_angle = clamped;

        std::ostringstream oss;
        oss << "#" << std::setw(3) << std::setfill('0') << _id
            << "P" << std::setw(4) << std::setfill('0') << _pwm
            << "T" << std::setw(4) << std::setfill('0') << _time_ms << "!";
        _cmd = oss.str();
    }
    const std::string& get_command_buffer() const{
        return _cmd;
    }
    void set_pwm_range(int min_pwm, int max_pwm){
        _min_pwm = std::clamp(min_pwm, 500, 2500);
        _max_pwm = std::clamp(max_pwm, 500, 2500);
        if (_min_pwm > _max_pwm) {
            std::swap(_min_pwm, _max_pwm);
        }
    }
private:
    int _id;
    float _angle;
    float _speed;
    float _last_angle{0.0f};
    int _pwm{1500};
    int _time_ms{0};
    int _min_pwm{500};
    int _max_pwm{2500};
    std::string _cmd;

    static int angle_to_pwm(float angle_deg, float min_angle_deg, float max_angle_deg, int min_pwm, int max_pwm){
        const float span = max_angle_deg - min_angle_deg;
        if (span <= 0.0f) return 1500;
        const float t = (angle_deg - min_angle_deg) / span;
        const float pwm = static_cast<float>(min_pwm) + t * static_cast<float>(max_pwm - min_pwm);
        return static_cast<int>(std::lround(pwm));
    }
};

class GimbalControl{
public:
    GimbalControl(){
        _pitch_motor.set_pwm_range(1200, 1800);
        std::cout << "Gimbal created." << std::endl;
    }
    void set_pitch_angle(float angle){
        _pitch_motor.set_angle(angle);
    }
    void set_yaw_angle(float angle){
        _yaw_motor.set_angle(angle);
    }
    void set_pitch_speed(float speed){
        _pitch_motor.set_speed(speed);
    }
    void set_yaw_speed(float speed){
        _yaw_motor.set_speed(speed);
    }
    void get_command(){
        _pitch_motor.generate_command(0.0f, 180.0f);
        _yaw_motor.generate_command(0.0f, 270.0f);
        _command_buffer = _pitch_motor.get_command_buffer() + _yaw_motor.get_command_buffer();
    }
    const std::string& get_command_buffer() const{
        return _command_buffer;
    }
    bool open_serial(const std::string& device, int baudrate){
        return _serial.open(device, baudrate);
    }
    void close_serial(){
        _serial.close();
    }
    bool is_serial_open() const{
        return _serial.is_open();
    }
    bool send_command(){
        if (!_serial.is_open()) {
            std::cout << "Sending command (serial closed): " << get_command_buffer() << std::endl;
            return false;
        }
        return _serial.write_string(get_command_buffer());
    }
    bool send_raw_command(const std::string& command){
        if (!_serial.is_open()) {
            std::cout << "Sending raw command (serial closed): " << command << std::endl;
            return false;
        }
        return _serial.write_string(command);
    }

private:
    ServoMotor _pitch_motor{3, 180.0f, 0.0f, 270.0f};
    ServoMotor _yaw_motor{0};
    std::string _command_buffer;
    SerialPort _serial;
};

#endif // GIMBAL_CONTROL_HPP