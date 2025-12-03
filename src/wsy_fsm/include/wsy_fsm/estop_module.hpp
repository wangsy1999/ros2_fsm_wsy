#pragma once

#include <rclcpp/rclcpp.hpp>
#include "interface_protocol/msg/joint_command.hpp"

class EstopModule
{
public:
    EstopModule(
        rclcpp::Publisher<interface_protocol::msg::JointCommand>::SharedPtr pub,
        int joint_num);

    // 在 ESTOP 状态中每 tick 调用 -> 持续松力
    void tick();

private:
    rclcpp::Publisher<interface_protocol::msg::JointCommand>::SharedPtr pub_;
    int joint_num_;
};
