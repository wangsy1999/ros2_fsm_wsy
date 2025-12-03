#pragma once

#include <vector>
#include <string>
#include <rclcpp/rclcpp.hpp>
#include "interface_protocol/msg/joint_state.hpp"
#include "interface_protocol/msg/joint_command.hpp"

class ResetModule
{
public:
    ResetModule(rclcpp::Node* node,
                rclcpp::Publisher<interface_protocol::msg::JointCommand>::SharedPtr pub,
                int joint_num,
                double duration_sec);

    // 从 YAML 加载 initial_pose / initial_kp / initial_kd
    bool loadYaml(const std::string& yaml_path);

    // 开始一次复位（记录起始姿态）
    bool begin(const interface_protocol::msg::JointState& start_state);

    // 在 RESET 状态中，每 tick 调用一次
    // 返回 true 表示本次复位完成
    bool tick(const interface_protocol::msg::JointState& current_state);

private:
    rclcpp::Node* node_;
    rclcpp::Publisher<interface_protocol::msg::JointCommand>::SharedPtr pub_;

    int joint_num_;
    double duration_sec_;
    rclcpp::Time start_time_;
    bool active_{false};

    std::vector<double> start_pos_;
    std::vector<double> target_pos_;
    std::vector<double> kp_;
    std::vector<double> kd_;
};
