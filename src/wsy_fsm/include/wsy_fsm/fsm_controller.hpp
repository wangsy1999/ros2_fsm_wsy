#pragma once

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include "interface_protocol/msg/joint_state.hpp"
#include "interface_protocol/msg/joint_command.hpp"

#include "wsy_fsm/reset_module.hpp"
#include "wsy_fsm/estop_module.hpp"
#include "wsy_fsm/run_module.hpp"
#include "wsy_fsm/keyboard.hpp"

// 四状态：Idle / Reset / Run / Estop
enum class ControlState {
    IDLE = 0,
    RESET = 1,
    RUN = 2,
    ESTOP = 3
};

inline std::string to_string(ControlState s)
{
    switch (s) {
        case ControlState::IDLE:  return "IDLE";
        case ControlState::RESET: return "RESET";
        case ControlState::RUN:   return "RUN";
        case ControlState::ESTOP: return "ESTOP";
    }
    return "UNKNOWN";
}

class FSMController : public rclcpp::Node
{
public:
    FSMController();
    ~FSMController();

private:
    // 回调
    void jointStateCallback(const interface_protocol::msg::JointState::SharedPtr msg);
    void fsmCmdCallback(const std_msgs::msg::String::SharedPtr msg);

    // 主循环（500Hz）
    void mainLoop();

    // 键盘
    void onKey(char c);

    // 状态切换
    void setState(ControlState new_state);
    void setStateByString(const std::string& s);

    // 工具
    bool hasValidState() const;

private:
    int joint_num_{0};
    double reset_duration_{2.0};   // s

    ControlState state_{ControlState::IDLE};
    
    // ROS 通信
    rclcpp::Subscription<interface_protocol::msg::JointState>::SharedPtr joint_state_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr fsm_cmd_sub_;
    rclcpp::Publisher<interface_protocol::msg::JointCommand>::SharedPtr joint_cmd_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    interface_protocol::msg::JointState last_state_;

    // 模块
    std::shared_ptr<ResetModule> reset_module_;
    std::shared_ptr<EstopModule> estop_module_;
    std::shared_ptr<RunModule>   run_module_;

    // 键盘
    KeyboardListener keyboard_;
};
