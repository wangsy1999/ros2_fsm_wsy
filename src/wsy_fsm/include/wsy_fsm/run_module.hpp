#pragma once

#include <rclcpp/rclcpp.hpp>
#include "interface_protocol/msg/joint_state.hpp"
#include "interface_protocol/msg/joint_command.hpp"

// 目前只是占位，留给你放 RL 推理
class RunModule
{
public:
    RunModule(
        rclcpp::Node* node,
        rclcpp::Publisher<interface_protocol::msg::JointCommand>::SharedPtr pub,
        int joint_num);

    // 在 RUN 状态中每 tick 调用
    void tick(const interface_protocol::msg::JointState& state);

private:
    rclcpp::Node* node_;
    rclcpp::Publisher<interface_protocol::msg::JointCommand>::SharedPtr pub_;
    int joint_num_;
};
