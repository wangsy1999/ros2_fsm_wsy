#include "wsy_fsm/run_module.hpp"

RunModule::RunModule(
    rclcpp::Node* node,
    rclcpp::Publisher<interface_protocol::msg::JointCommand>::SharedPtr pub,
    int joint_num)
: node_(node)
, pub_(pub)
, joint_num_(joint_num)
{
}

void RunModule::tick(const interface_protocol::msg::JointState& state)
{
    // 这里现在是空壳，给你预留 RL 推理接口位置。
    // 你未来可以这样做：
    //
    // 1. 从 state 中提取观测
    // 2. 调用 OpenVINO / LibTorch 推理
    // 3. 填充 JointCommand 并发布
    //
    // 示例：只打印一次，避免刷屏
    static bool printed = false;
    if (!printed) {
        RCLCPP_INFO(node_->get_logger(),
                    "RunModule::tick called. RL policy not implemented yet.");
        printed = true;
    }
    (void)state;
}
