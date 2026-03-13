#include "TargetTracking/SerialPort.hpp"

#ifdef MYGO_TARGETTRACKING_USE_MAIX

#include "maix_uart.hpp"

class SerialPort::MaixUartImpl {
public:
    std::unique_ptr<maix::peripheral::uart::UART> uart;
    int baudrate = 115200;
};

SerialPort::SerialPort() : uart_impl_(std::make_unique<MaixUartImpl>()) {}

SerialPort::~SerialPort() {
    close();
}

bool SerialPort::open(const std::string& device, int baudrate) {
    close();
    device_ = device;
    uart_impl_->baudrate = baudrate;
    uart_impl_->uart = std::make_unique<maix::peripheral::uart::UART>(
        device,
        baudrate,
        maix::peripheral::uart::BITS_8,
        maix::peripheral::uart::PARITY_NONE,
        maix::peripheral::uart::STOP_1,
        maix::peripheral::uart::FLOW_CTRL_NONE
    );
    return uart_impl_->uart != nullptr && uart_impl_->uart->is_open();
}

void SerialPort::close() {
    if (uart_impl_ && uart_impl_->uart) {
        uart_impl_->uart->close();
        uart_impl_->uart.reset();
    }
}

bool SerialPort::is_open() const {
    return uart_impl_ && uart_impl_->uart && uart_impl_->uart->is_open();
}

bool SerialPort::set_baudrate(int baudrate) {
    if (!uart_impl_) {
        return false;
    }
    uart_impl_->baudrate = baudrate;
    if (!is_open()) {
        return true;
    }
    return open(device_, baudrate);
}

bool SerialPort::write_bytes(const uint8_t* data, size_t size) {
    if (!is_open() || data == nullptr || size == 0) {
        return false;
    }
    return uart_impl_->uart->write(data, static_cast<int>(size)) >= 0;
}

bool SerialPort::write_string(const std::string& data) {
    if (!is_open() || data.empty()) {
        return false;
    }
    return uart_impl_->uart->write(data) >= 0;
}

bool SerialPort::configure_port(int baudrate) {
    return set_baudrate(baudrate);
}

int SerialPort::to_termios_baud(int baudrate) {
    return baudrate;
}

#else

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

#endif
