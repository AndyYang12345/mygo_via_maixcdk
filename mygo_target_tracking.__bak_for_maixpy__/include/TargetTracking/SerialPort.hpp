#ifndef SERIAL_PORT_HPP
#define SERIAL_PORT_HPP

#include <cstdint>
#include <memory>
#include <string>

class SerialPort {
public:
    SerialPort();
    ~SerialPort();

    bool open(const std::string& device, int baudrate);
    void close();
    bool is_open() const;

    bool set_baudrate(int baudrate);

    bool write_bytes(const uint8_t* data, size_t size);
    bool write_string(const std::string& data);

private:
#ifdef MYGO_TARGETTRACKING_USE_MAIX
    class MaixUartImpl;
    std::unique_ptr<MaixUartImpl> uart_impl_;
#else
    int fd_;
#endif
    std::string device_;

    bool configure_port(int baudrate);
    static int to_termios_baud(int baudrate);
};

#endif // SERIAL_PORT_HPP
