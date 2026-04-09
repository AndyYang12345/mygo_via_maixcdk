#ifndef SERIAL_PORT_HPP
#define SERIAL_PORT_HPP

#include <cstdint>
#include <string>

class SerialPort {
public:
    /// 构造串口对象，初始为未打开状态。
    SerialPort();
    /// 析构时自动关闭串口句柄。
    ~SerialPort();

    /// 打开指定设备并配置波特率。
    bool open(const std::string& device, int baudrate);
    /// 关闭串口（可重复调用）。
    void close();
    /// 判断串口句柄当前是否有效。
    bool is_open() const;

    /// 在已打开状态下动态设置波特率。
    bool set_baudrate(int baudrate);

    /// 写入原始字节流，内部处理部分写入重试。
    bool write_bytes(const uint8_t* data, size_t size);
    /// 写入字符串数据（等价于字节写入包装）。
    bool write_string(const std::string& data);

private:
    int fd_;
    std::string device_;

    /// 使用 termios 参数初始化串口工作模式。
    bool configure_port(int baudrate);
    /// 将整型波特率映射为 termios 常量。
    static int to_termios_baud(int baudrate);
};

#endif // SERIAL_PORT_HPP
