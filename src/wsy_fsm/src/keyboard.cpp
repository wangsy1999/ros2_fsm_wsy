#include "wsy_fsm/keyboard.hpp"
#include <termios.h>
#include <unistd.h>
#include <chrono>

namespace {

// 非阻塞 getch
char getch_nonblock()
{
    char buf = 0;
    termios old = {0};
    tcgetattr(STDIN_FILENO, &old);

    termios newt = old;
    newt.c_lflag &= ~ICANON;
    newt.c_lflag &= ~ECHO;
    newt.c_cc[VMIN] = 0;
    newt.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);

    int nread = read(STDIN_FILENO, &buf, 1);
    tcsetattr(STDIN_FILENO, TCSANOW, &old);

    if (nread == 1) return buf;
    return 0;
}

} // namespace

KeyboardListener::KeyboardListener() {}

KeyboardListener::~KeyboardListener()
{
    stop();
}

void KeyboardListener::start(std::function<void(char)> callback)
{
    callback_ = callback;
    running_ = true;
    thread_ = std::thread(&KeyboardListener::loop, this);
}

void KeyboardListener::stop()
{
    running_ = false;
    if (thread_.joinable()) {
        thread_.join();
    }
}

void KeyboardListener::loop()
{
    while (running_) {
        char c = getch_nonblock();
        if (c != 0 && callback_) {
            callback_(c);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}
