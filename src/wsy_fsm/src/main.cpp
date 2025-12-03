#include <rclcpp/rclcpp.hpp>
#include "wsy_fsm/fsm_controller.hpp"

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<FSMController>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
