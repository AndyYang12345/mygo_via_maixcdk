#include "TargetTracking/SerialPort.hpp"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

SerialPort::SerialPort() : fd_(-1) {}

SerialPort::~SerialPort() {
    close();
}

bool SerialPort::open(const std::string& device, int baudrate) {
    close();
    device_ = device;
    fd_ = ::open(device.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
    if (fd_ < 0) {
        return false;
    }
    return configure_port(baudrate);
}

void SerialPort::close() {
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

bool SerialPort::is_open() const {
    return fd_ >= 0;
}

bool SerialPort::set_baudrate(int baudrate) {
    if (fd_ < 0) {
        return false;
    }
    return configure_port(baudrate);
}

bool SerialPort::write_bytes(const uint8_t* data, size_t size) {
    if (fd_ < 0 || data == nullptr || size == 0) {
        return false;
    }
    ssize_t total_written = 0;
    while (total_written < static_cast<ssize_t>(size)) {
        ssize_t written = ::write(fd_, data + total_written, size - total_written);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        total_written += written;
    }
    return true;
}

bool SerialPort::write_string(const std::string& data) {
    return write_bytes(reinterpret_cast<const uint8_t*>(data.data()), data.size());
}

bool SerialPort::configure_port(int baudrate) {
    termios tty{};
    if (tcgetattr(fd_, &tty) != 0) {
        return false;
    }

    cfmakeraw(&tty);

    int baud = to_termios_baud(baudrate);
    if (baud == 0) {
        return false;
    }

    cfsetispeed(&tty, baud);
    cfsetospeed(&tty, baud);

    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CRTSCTS;
    tty.c_cflag &= ~PARENB;
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;

    tty.c_cc[VMIN] = 0;
    tty.c_cc[VTIME] = 5;

    if (tcsetattr(fd_, TCSANOW, &tty) != 0) {
        return false;
    }
    return true;
}

int SerialPort::to_termios_baud(int baudrate) {
    switch (baudrate) {
        case 9600: return B9600;
        case 19200: return B19200;
        case 38400: return B38400;
        case 57600: return B57600;
        case 115200: return B115200;
        case 230400: return B230400;
        default: return 0;
    }
}
