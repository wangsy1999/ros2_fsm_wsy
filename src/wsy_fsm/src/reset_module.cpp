#include "wsy_fsm/reset_module.hpp"
#include <yaml-cpp/yaml.h>

ResetModule::ResetModule(
    rclcpp::Node* node,
    rclcpp::Publisher<interface_protocol::msg::JointCommand>::SharedPtr pub,
    int joint_num,
    double duration_sec)
: node_(node)
, pub_(pub)
, joint_num_(joint_num)
, duration_sec_(duration_sec)
{
    start_pos_.resize(joint_num_, 0.0);
    target_pos_.resize(joint_num_, 0.0);
    kp_.resize(joint_num_, 0.0);
    kd_.resize(joint_num_, 0.0);
}

bool ResetModule::loadYaml(const std::string& yaml_path)
{
    try {
        YAML::Node cfg = YAML::LoadFile(yaml_path);
        target_pos_ = cfg["initial_pose"].as<std::vector<double>>();
        kp_         = cfg["initial_kp"].as<std::vector<double>>();
        kd_         = cfg["initial_kd"].as<std::vector<double>>();

        if ((int)target_pos_.size() != joint_num_ ||
            (int)kp_.size() != joint_num_ ||
            (int)kd_.size() != joint_num_) {
            RCLCPP_ERROR(node_->get_logger(),
                         "ResetModule: YAML size mismatch. pose=%zu kp=%zu kd=%zu, expected=%d",
                         target_pos_.size(), kp_.size(), kd_.size(), joint_num_);
            return false;
        }

        RCLCPP_INFO(node_->get_logger(), "ResetModule: YAML loaded from %s", yaml_path.c_str());
        return true;
    }
    catch (const std::exception& e) {
        RCLCPP_ERROR(node_->get_logger(),
                     "ResetModule: failed to load YAML: %s", e.what());
        return false;
    }
}

bool ResetModule::begin(const interface_protocol::msg::JointState& start_state)
{
    if ((int)start_state.position.size() != joint_num_) {
        RCLCPP_WARN(node_->get_logger(),
                    "ResetModule::begin: invalid start_state size = %zu, expect = %d",
                    start_state.position.size(), joint_num_);
        return false;
    }

    start_pos_ = start_state.position;
    start_time_ = node_->now();
    active_ = true;
    return true;
}

bool ResetModule::tick(const interface_protocol::msg::JointState& current_state)
{
    if (!active_) {
        return true;  // 不在 reset 中，视为已完成
    }

    const double publish_hz = 500.0; // 实际发送频率
    (void)publish_hz;  // 由上层 Timer 控制频率

    rclcpp::Time now = node_->now();
    double elapsed = (now - start_time_).seconds();
    double a = elapsed / duration_sec_;
    if (a > 1.0) a = 1.0;
    if (a < 0.0) a = 0.0;

    interface_protocol::msg::JointCommand cmd;
    cmd.position.resize(joint_num_);
    cmd.velocity.resize(joint_num_);
    cmd.torque.resize(joint_num_);
    cmd.feed_forward_torque.resize(joint_num_);
    cmd.stiffness.resize(joint_num_);
    cmd.damping.resize(joint_num_);
    cmd.parallel_parser_type = 0;

    for (int i = 0; i < joint_num_; i++) {
        double p0 = start_pos_[i];
        double p1 = target_pos_[i];
        cmd.position[i] = p0 * (1.0 - a) + p1 * a;
        cmd.velocity[i] = 0.0;
        cmd.torque[i]   = 0.0;
        cmd.feed_forward_torque[i] = 0.0;
        cmd.stiffness[i] = kp_[i];
        cmd.damping[i]   = kd_[i];
    }

    pub_->publish(cmd);

    if (a >= 1.0) {
        active_ = false;
        return true;  // 完成
    }
    return false;     // 还在进行中
}
