#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

#include "interface_protocol/msg/joint_state.hpp"
#include "interface_protocol/msg/joint_command.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>

#include <atomic>
#include <thread>
#include <termios.h>
#include <unistd.h>

#include <yaml-cpp/yaml.h>

// ===================================================
// 全局变量：Ctrl+C 时松力
// ===================================================
static std::atomic<bool> g_exiting(false);
static rclcpp::Publisher<interface_protocol::msg::JointCommand>::SharedPtr g_pub;
static int g_joint_num = 0;

// ===================================================
// 非阻塞按键读取
// ===================================================
char getch_nonblock()
{
    char buf = 0;
    termios old = {0};
    tcgetattr( STDIN_FILENO, &old );

    termios newt = old;
    newt.c_lflag &= ~ICANON;
    newt.c_lflag &= ~ECHO;
    newt.c_cc[VMIN] = 0;
    newt.c_cc[VTIME] = 0;

    tcsetattr( STDIN_FILENO, TCSANOW, &newt );
    int nread = read( STDIN_FILENO, &buf, 1 );
    tcsetattr( STDIN_FILENO, TCSANOW, &old );

    if (nread == 1) return buf;
    return 0;
}

// ===================================================
// Ctrl+C 强制松力（连发 50 帧）
// ===================================================
void sigint_force_relax(int)
{
    if (g_exiting.exchange(true)) return;

    if (g_pub) {
        interface_protocol::msg::JointCommand msg;
        msg.position.resize(g_joint_num, 0.0);
        msg.velocity.resize(g_joint_num, 0.0);
        msg.torque.resize(g_joint_num, 0.0);
        msg.feed_forward_torque.resize(g_joint_num, 0.0);
        msg.stiffness.resize(g_joint_num, 0.0);
        msg.damping.resize(g_joint_num, 0.0);
        msg.parallel_parser_type = 0;

        for (int i = 0; i < 50; i++) {
            g_pub->publish(msg);
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }

    rclcpp::shutdown();
}

// ===================================================
// 节点定义
// ===================================================
class SimpleResetNode : public rclcpp::Node
{
public:
    SimpleResetNode()
    : Node("simple_motor_reset_node")
    {
        joint_num_ = declare_parameter<int>("joint_num", 24);
        duration_sec_ = declare_parameter<double>("reset_duration", 3.0);
        steps_        = declare_parameter<int>("reset_steps", 1000);

        // YAML 文件读取
        std::string config_file =
            declare_parameter<std::string>("config_file", "reset_pose.yaml");


        std::string yaml_path = "/home/wsy/rl/ros2_ws/src/motor_reset/config/reset_pose.yaml";

        RCLCPP_INFO(get_logger(), "Loading reset YAML: %s", yaml_path.c_str());

        YAML::Node config = YAML::LoadFile(yaml_path);

        yaml_pose_ = config["initial_pose"].as<std::vector<double>>();
        yaml_kp_   = config["initial_kp"].as<std::vector<double>>();
        yaml_kd_   = config["initial_kd"].as<std::vector<double>>();

        if (yaml_pose_.size() != joint_num_) {
            RCLCPP_FATAL(get_logger(), "initial_pose size mismatch!");
        }

        cmd_msg_ = std::make_shared<interface_protocol::msg::JointCommand>();
        resize_joint_command();

        // 订阅状态
        joint_state_sub_ = create_subscription<interface_protocol::msg::JointState>(
            "/hardware/joint_state",
            10,
            std::bind(&SimpleResetNode::jointStateCallback, this, std::placeholders::_1)
        );

        // 发布命令
        joint_cmd_pub_ = create_publisher<interface_protocol::msg::JointCommand>(
            "/hardware/joint_command",
            rclcpp::QoS(1).best_effort()
        );


        // reset_cmd topic 仍然支持
        reset_sub_ = create_subscription<std_msgs::msg::String>(
            "/reset_cmd",
            10,
            std::bind(&SimpleResetNode::resetCallback, this, std::placeholders::_1)
        );

        // Ctrl-C 松力
        g_pub = joint_cmd_pub_;
        g_joint_num = joint_num_;

        // 键盘监听线程
        running_ = true;
        keyboard_thread_ = std::thread(&SimpleResetNode::keyboardLoop, this);

        RCLCPP_INFO(get_logger(), "Reset node ready. Keys: [q=relax] [p=pose]");
    }

    ~SimpleResetNode()
    {
        running_ = false;
        if (keyboard_thread_.joinable())
            keyboard_thread_.join();
    }

private:

    // ======================================================
    // 键盘监听：q / r / z / p
    // ======================================================
    void keyboardLoop()
    {
        while (running_) {
            char c = getch_nonblock();
            if (c == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                continue;
            }

            switch (c) {
                case 'q':
                    RCLCPP_WARN(get_logger(), "Keyboard relax");
                    relax_all();
                    break;

                case 'p':
                    RCLCPP_INFO(get_logger(), "Keyboard pose");
                    reset_to_initial_pose();
                    break;
            }
        }
    }

    // ======================================================
    // 松力
    // ======================================================
    void relax_all()
    {
        interface_protocol::msg::JointCommand msg;
        msg.position.resize(joint_num_, 0.0);
        msg.velocity.resize(joint_num_, 0.0);
        msg.torque.resize(joint_num_, 0.0);
        msg.feed_forward_torque.resize(joint_num_, 0.0);
        msg.stiffness.resize(joint_num_, 0.0);
        msg.damping.resize(joint_num_, 0.0);
        msg.parallel_parser_type = 0;

        rclcpp::WallRate rate(200);
        for (int i = 0; i < 20; i++) {
            joint_cmd_pub_->publish(msg);
            rate.sleep();
        }
    }

    // ======================================================
    // Reset CMD 回调
    // ======================================================
    void resetCallback(const std_msgs::msg::String::SharedPtr msg)
    {

        if (msg->data == "pose") reset_to_initial_pose();
    }

    // ======================================================
    // JointState 回调
    // ======================================================
    void jointStateCallback(const interface_protocol::msg::JointState::SharedPtr msg)
    {
        last_state_ = *msg;
    }

    // ======================================================
    // Reset 动作
    // ======================================================

    // YAML pose + PD
    void reset_to_initial_pose()
    {
        if (!valid()) return;

        smooth_interpolate_with_pd(
            last_state_.position,
            yaml_pose_,
            yaml_kp_,
            yaml_kd_);
    }

    bool valid()
    {
        if (last_state_.position.size() != joint_num_) {
            RCLCPP_WARN(get_logger(), "Invalid joint_state");
            return false;
        }
        return true;
    }

    // ======================================================
    // 插值（YAML 中每关节的 PD）
    // ======================================================
    void smooth_interpolate_with_pd(
        const std::vector<double>& start,
        const std::vector<double>& target,
        const std::vector<double>& kp,
        const std::vector<double>& kd)
    {
        const double publish_hz = 500.0;
        rclcpp::Rate rate(publish_hz);

        int total_steps = static_cast<int>(duration_sec_ * publish_hz);

        for (int s = 0; s <= total_steps; s++) 
        {
            double a = double(s) / double(total_steps);

            for (int i = 0; i < joint_num_; i++) {

                cmd_msg_->position[i] =
                    start[i] * (1.0 - a) + target[i] * a;

                cmd_msg_->velocity[i] = 0;
                cmd_msg_->torque[i]   = 0;
                cmd_msg_->feed_forward_torque[i] = 0;
                cmd_msg_->stiffness[i] = kp[i];
                cmd_msg_->damping[i]   = kd[i];
            }

            cmd_msg_->parallel_parser_type = 0;
            joint_cmd_pub_->publish(*cmd_msg_);

            rate.sleep();
        }
    }

    void resize_joint_command()
    {
        cmd_msg_->position.resize(joint_num_);
        cmd_msg_->velocity.resize(joint_num_);
        cmd_msg_->torque.resize(joint_num_);
        cmd_msg_->feed_forward_torque.resize(joint_num_);
        cmd_msg_->stiffness.resize(joint_num_);
        cmd_msg_->damping.resize(joint_num_);
    }

private:

    int joint_num_;
    double duration_sec_;
    int steps_;

    bool running_ = false;
    std::thread keyboard_thread_;

    interface_protocol::msg::JointState last_state_;
    std::shared_ptr<interface_protocol::msg::JointCommand> cmd_msg_;

    rclcpp::Subscription<interface_protocol::msg::JointState>::SharedPtr joint_state_sub_;
    rclcpp::Publisher<interface_protocol::msg::JointCommand>::SharedPtr joint_cmd_pub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr reset_sub_;

    // YAML data
    std::vector<double> yaml_pose_;
    std::vector<double> yaml_kp_;
    std::vector<double> yaml_kd_;
};

// ======================================================
// main()
// ======================================================
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    signal(SIGINT, sigint_force_relax);

    auto node = std::make_shared<SimpleResetNode>();

    rclcpp::spin(node);

    rclcpp::shutdown();
    return 0;
}
