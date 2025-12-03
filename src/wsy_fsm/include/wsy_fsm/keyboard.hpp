#pragma once

#include <thread>
#include <atomic>
#include <functional>

class KeyboardListener
{
public:
    KeyboardListener();
    ~KeyboardListener();

    // callback(char key)
    void start(std::function<void(char)> callback);
    void stop();

private:
    void loop();

private:
    std::atomic<bool> running_{false};
    std::thread thread_;
    std::function<void(char)> callback_;
};
