#include "wsy_fsm/estop_module.hpp"

EstopModule::EstopModule(
    rclcpp::Publisher<interface_protocol::msg::JointCommand>::SharedPtr pub,
    int joint_num)
: pub_(pub)
, joint_num_(joint_num)
{
}

void EstopModule::tick()
{
    interface_protocol::msg::JointCommand cmd;
    cmd.position.resize(joint_num_, 0.0);
    cmd.velocity.resize(joint_num_, 0.0);
    cmd.torque.resize(joint_num_, 0.0);
    cmd.feed_forward_torque.resize(joint_num_, 0.0);
    cmd.stiffness.resize(joint_num_, 0.0);
    cmd.damping.resize(joint_num_, 0.0);
    cmd.parallel_parser_type = 0;

    pub_->publish(cmd);
}
