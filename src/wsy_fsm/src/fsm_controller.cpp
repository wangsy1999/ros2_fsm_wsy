#include "wsy_fsm/fsm_controller.hpp"
#include <ament_index_cpp/get_package_share_directory.hpp>

FSMController::FSMController()
: Node("fsm_controller")
{
    joint_num_ = declare_parameter<int>("joint_num", 24);
    reset_duration_ = declare_parameter<double>("reset_duration", 2.0);
    std::string yaml_file = declare_parameter<std::string>("config_file", "reset_pose.yaml");

    // JointCommand Publisher（控制）
    auto control_qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
    joint_cmd_pub_ = create_publisher<interface_protocol::msg::JointCommand>(
        "/hardware/joint_command", control_qos);

    // JointState Subscriber（传感器）
    auto sensor_qos = rclcpp::SensorDataQoS();
    joint_state_sub_ = create_subscription<interface_protocol::msg::JointState>(
        "/hardware/joint_state",
        sensor_qos,
        std::bind(&FSMController::jointStateCallback, this, std::placeholders::_1)
    );

    // FSM 控制 Topic
    auto reliable_qos = rclcpp::QoS(10).reliable();
    fsm_cmd_sub_ = create_subscription<std_msgs::msg::String>(
        "/fsm_cmd",
        reliable_qos,
        std::bind(&FSMController::fsmCmdCallback, this, std::placeholders::_1)
    );

    // 创建模块
    reset_module_ = std::make_shared<ResetModule>(this, joint_cmd_pub_, joint_num_, reset_duration_);
    estop_module_ = std::make_shared<EstopModule>(joint_cmd_pub_, joint_num_);
    run_module_   = std::make_shared<RunModule>(this, joint_cmd_pub_, joint_num_);
    run_module_->loadYaml("/home/wsy/rl/ros2_ws/src/wsy_fsm/config/reset_pose.yaml");
    if (!run_module_->init()) {
        RCLCPP_ERROR(get_logger(), "RunModule init() failed.");
    }
    // 加载 YAML
    try {
        std::string yaml_path = "/home/wsy/rl/ros2_ws/src/wsy_fsm/config/reset_pose.yaml";
        if (!reset_module_->loadYaml(yaml_path)) {
            RCLCPP_ERROR(get_logger(), "FSMController: failed to load reset pose YAML.");
        }
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "FSMController: get_package_share_directory failed: %s", e.what());
    }

    // 定时器 500Hz
    timer_ = create_wall_timer(
        std::chrono::milliseconds(2),
        std::bind(&FSMController::mainLoop, this)
    );

    // 键盘：i=IDLE, r=RESET, p=RUN, e=ESTOP
    keyboard_.start([this](char c){ this->onKey(c); });

    state_ = ControlState::IDLE;
    RCLCPP_INFO(get_logger(), "FSMController started. State = IDLE");
}

FSMController::~FSMController()
{
    keyboard_.stop();
}

void FSMController::jointStateCallback(const interface_protocol::msg::JointState::SharedPtr msg)
{
    last_state_ = *msg;
}

void FSMController::fsmCmdCallback(const std_msgs::msg::String::SharedPtr msg)
{
    setStateByString(msg->data);
}

void FSMController::mainLoop()
{    // ======================================================
    // Fake Power Control
    // ======================================================
    if (!power_on_) {
        // 发布松力命令，但保持 last_state_ 的 position 不变或置 0
        interface_protocol::msg::JointCommand cmd;
        cmd.position = (last_state_.position.size() == joint_num_)
                        ? last_state_.position
                        : std::vector<double>(joint_num_, 0.0);

        cmd.velocity.resize(joint_num_, 0.0);
        cmd.torque.resize(joint_num_, 0.0);
        cmd.feed_forward_torque.resize(joint_num_, 0.0);
        cmd.stiffness.resize(joint_num_, 0.0);
        cmd.damping.resize(joint_num_, 0.0);
        cmd.parallel_parser_type = 0;

        joint_cmd_pub_->publish(cmd);
        return;   // 所有 RESET/RUN/ESTOP 均不执行
    }
    switch (state_) {
        case ControlState::IDLE:
            // 不做任何控制
            break;

        case ControlState::RESET:
        {
            if (!hasValidState()) {
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                                     "RESET state: waiting for valid joint_state.");
                return;
            }
            // 第一次进入 RESET 时调用 begin
            static bool reset_started = false;
            if (!reset_started) {
                if (!reset_module_->begin(last_state_)) {
                    RCLCPP_ERROR(get_logger(), "ResetModule begin failed, fallback to IDLE.");
                    state_ = ControlState::IDLE;
                    return;
                }
                RCLCPP_WARN(get_logger(), "RESET state started.");
                reset_started = true;
            }
            bool done = reset_module_->tick(last_state_);
            if (done) {
                RCLCPP_INFO(get_logger(), "RESET finished -> IDLE");
                reset_started = false;
                state_ = ControlState::IDLE;
            }
            break;
        }

        case ControlState::RUN:
        {
            if (!hasValidState()) {
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                                     "RUN state: waiting for valid joint_state.");
                return;
            }
            run_module_->tick(last_state_);
            break;
        }

        case ControlState::ESTOP:
        {
            estop_module_->tick();
            break;
        }
    }
}

void FSMController::onKey(char c)
{
    switch (c) {
        case 'i':
            setState(ControlState::IDLE);
            break;
        case 'p':
            setState(ControlState::RESET);
            break;
        case 'r':
            setState(ControlState::RUN);
            break;
        case 'e':
            setState(ControlState::ESTOP);
            break;
        case '9':   // o = toggle power
            power_on_ = !power_on_;
            RCLCPP_WARN(get_logger(), "POWER BUTTON: %s",
                        power_on_ ? "ON (PD active)" : "OFF (PD=0)");
            break;

        default:
            break;
    }
}

void FSMController::setState(ControlState new_state)
{
    if (new_state == state_) return;

    RCLCPP_WARN(get_logger(), "FSM: %s -> %s",
                to_string(state_).c_str(),
                to_string(new_state).c_str());
    state_ = new_state;
}

void FSMController::setStateByString(const std::string& s)
{
    if (s == "idle")  setState(ControlState::IDLE);
    if (s == "reset") setState(ControlState::RESET);
    if (s == "run")   setState(ControlState::RUN);
    if (s == "estop") setState(ControlState::ESTOP);
}

bool FSMController::hasValidState() const
{
    return ((int)last_state_.position.size() == joint_num_);
}
