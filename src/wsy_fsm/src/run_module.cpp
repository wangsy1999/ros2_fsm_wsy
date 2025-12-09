#include "wsy_fsm/run_module.hpp"

#include <chrono>
#include <iostream>

// =====================================================
// 关节映射： policy index → hardware index
// !!! 按你的实际机器人顺序修改下面这一行 !!!
// =====================================================
static constexpr std::array<int, 12> kPolicyToHw = {
    0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11
};

// =====================================================
// RunModule 构造
// =====================================================
RunModule::RunModule(
    rclcpp::Node* node,
    rclcpp::Publisher<interface_protocol::msg::JointCommand>::SharedPtr pub,
    int joint_num)
: node_(node)
, pub_(pub)
, joint_num_(12)  // 不再写死 12
, env_(ORT_LOGGING_LEVEL_WARNING, "fsm_run")
, session_options_{}
, memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    session_options_.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetExecutionMode(ORT_SEQUENTIAL);

    action_dim_ = static_cast<size_t>(joint_num_);

    // if (!load_params_from_ros()) {
    //     RCLCPP_ERROR(node_->get_logger(), "[RunModule] load_params_from_ros failed.");
    //     model_loaded_ = false;
    //     return;
    // }

    FrameTensor zero_frame;
    zero_frame.data.assign(frame_dim_, 0.0f);
    history_.init(static_cast<size_t>(input_stack_len_), zero_frame);

    last_action_.assign(joint_num_, 0.0f);

    // if (!init_onnx_session()) {
    //     model_loaded_ = false;
    //     return;
    // }

//     model_loaded_ = true;
//     RCLCPP_INFO(node_->get_logger(),
//         "[RunModule] Init OK. model=%s, input_stack=%d, frame_dim=%zu, obs_dim=%zu, action_dim=%zu",
//         onnx_model_path_.c_str(), input_stack_len_, frame_dim_, obs_dim_, action_dim_);
}
#include <yaml-cpp/yaml.h>
bool RunModule::init()
{
    // 需要 YAML 已经加载
    if (onnx_model_path_.empty()) {
        RCLCPP_ERROR(node_->get_logger(), "[RunModule] init(): YAML not loaded or missing model path");
        return false;
    }

    // 计算 frame_dim / obs_dim
    frame_dim_ =
          3  // ang_vel
        + 3  // proj gravity
        + 3  // cmd
        + 2  // clock
        + joint_num_     // q
        + joint_num_     // dq
        + joint_num_;    // last_action

    obs_dim_ = static_cast<size_t>(input_stack_len_) * frame_dim_;

    // 初始化 history buffer
    FrameTensor zero_frame;
    zero_frame.data.assign(frame_dim_, 0.0f);
    history_.init(static_cast<size_t>(input_stack_len_), zero_frame);

    last_action_.assign(joint_num_, 0.0f);

    // 初始化 ONNX SESSION
    if (!init_onnx_session()) {
        RCLCPP_ERROR(node_->get_logger(), "[RunModule] init(): failed to create ONNX session");
        model_loaded_ = false;
        return false;
    }

    model_loaded_ = true;

    RCLCPP_INFO(node_->get_logger(),
        "[RunModule] init OK. frame_dim=%zu obs_dim=%zu", frame_dim_, obs_dim_);

    return true;
}

bool RunModule::loadYaml(const std::string& yaml_path)
{
    YAML::Node config;

    try {
        config = YAML::LoadFile(yaml_path);
    }
    catch (const std::exception& e) {
        RCLCPP_ERROR(node_->get_logger(),
                     "[RunModule] Failed to load YAML: %s", e.what());
        return false;
    }

    // --- ONNX 路径 ---
    if (config["onnx_model_path"])
        onnx_model_path_ = config["onnx_model_path"].as<std::string>();
if (config["input_node_name"])
    input_node_name_ = config["input_node_name"].as<std::string>();

if (config["output_node_name"])
    output_node_name_ = config["output_node_name"].as<std::string>();

    // --- stack & timing ---
    if (config["input_stack_len"])
        input_stack_len_ = config["input_stack_len"].as<int>();

    if (config["cycle_time"])
        cycle_time_ = config["cycle_time"].as<double>();

    if (config["control_dt"])
        dt_ = config["control_dt"].as<double>();

    // --- Action parameters (12 DOF) ---
    if (config["action_scale"])
        action_scale_ = config["action_scale"].as<std::vector<Float>>();

    if (config["default_position"])
        default_position_ = config["default_position"].as<std::vector<Float>>();

    if (config["joint_clip_upper"])
        joint_clip_upper_ = config["joint_clip_upper"].as<std::vector<Float>>();

    if (config["joint_clip_lower"])
        joint_clip_lower_ = config["joint_clip_lower"].as<std::vector<Float>>();

    if (config["run_kp"])
        run_kp_ = config["run_kp"].as<std::vector<double>>();

    if (config["run_kd"])
        run_kd_ = config["run_kd"].as<std::vector<double>>();

    RCLCPP_INFO(node_->get_logger(),
                "[RunModule] Successfully loaded YAML: %s",
                yaml_path.c_str());

    return true;
}

// =====================================================
// 从 ROS 参数加载
// =====================================================
// bool RunModule::load_params_from_ros()
// {
//     onnx_model_path_ = node_->declare_parameter<std::string>(
//         "onnx_model_path",
//         "/home/wsy/rl/LeggedLab_private/logs/pm01_flat/2025-12-08_15-40-07/exported/policy.onnx");

//     input_stack_len_ = node_->declare_parameter<int>("input_stack_len", 15);
//     cycle_time_      = node_->declare_parameter<double>("cycle_time", 0.8);
//     dt_              = node_->declare_parameter<double>("control_dt", 0.02);

//     input_node_name_  = node_->declare_parameter<std::string>("input_node_name", "obs");
//     output_node_name_ = node_->declare_parameter<std::string>("output_node_name", "actions");

//     obs_clip_ = static_cast<Float>(node_->declare_parameter<double>("obs_clip", 100.0));
//     act_clip_ = static_cast<Float>(node_->declare_parameter<double>("act_clip", 1.0));

//     scale_ang_vel_         = static_cast<Float>(node_->declare_parameter<double>("scale_ang_vel", 1.0));
//     scale_project_gravity_ = static_cast<Float>(node_->declare_parameter<double>("scale_project_gravity", 1.0));
//     scale_cmd_             = static_cast<Float>(node_->declare_parameter<double>("scale_cmd", 1.0));
//     scale_dof_pos_         = static_cast<Float>(node_->declare_parameter<double>("scale_dof_pos", 1.0));
//     scale_dof_vel_         = static_cast<Float>(node_->declare_parameter<double>("scale_dof_vel", 0.005));
//     scale_last_action_     = static_cast<Float>(node_->declare_parameter<double>("scale_last_action", 1.0));

//     auto vec_default = std::vector<double>(joint_num_, 0.0);
//     auto vec_one     = std::vector<double>(joint_num_, 1.0);

//     auto action_scale_d = node_->declare_parameter<std::vector<double>>("action_scale", vec_one);
//     auto default_pos_d  = node_->declare_parameter<std::vector<double>>("default_position", vec_default);
//     auto joint_upper_d  = node_->declare_parameter<std::vector<double>>("joint_clip_upper", vec_one);
//     auto joint_lower_d  = node_->declare_parameter<std::vector<double>>("joint_clip_lower", vec_default);

//     if ((int)action_scale_d.size() != joint_num_ ||
//         (int)default_pos_d.size()  != joint_num_ ||
//         (int)joint_upper_d.size()  != joint_num_ ||
//         (int)joint_lower_d.size()  != joint_num_) {
//         RCLCPP_ERROR(node_->get_logger(), "[RunModule] YAML vector size mismatch.");
//         return false;
//     }

//     action_scale_.resize(joint_num_);
//     default_position_.resize(joint_num_);
//     joint_clip_upper_.resize(joint_num_);
//     joint_clip_lower_.resize(joint_num_);

//     for (int i = 0; i < joint_num_; i++) {
//         action_scale_[i]     = static_cast<Float>(action_scale_d[i]);
//         default_position_[i] = static_cast<Float>(default_pos_d[i]);
//         joint_clip_upper_[i] = static_cast<Float>(joint_upper_d[i]);
//         joint_clip_lower_[i] = static_cast<Float>(joint_lower_d[i]);
//     }

//     run_kp_ = node_->declare_parameter<std::vector<double>>("run_kp", std::vector<double>(joint_num_, 40.0));
//     run_kd_ = node_->declare_parameter<std::vector<double>>("run_kd", std::vector<double>(joint_num_, 1.0));

//     // 单帧维度
//     frame_dim_ =
//               3         // ang_vel
//             + 3         // proj_gravity
//             + 3         // cmd
//             + 2         // clock
//             + joint_num_ // q
//             + joint_num_ // dq
//             + joint_num_; // last action

//     obs_dim_ = frame_dim_ * input_stack_len_;
//     RCLCPP_WARN(node_->get_logger(),
//         "[RunModule] YAML loaded: action_scale[0]=%.3f  default_position[0]=%.3f  joint_clip_upper[0]=%.3f",
//         action_scale_[0], default_position_[0], joint_clip_upper_[0]);

//     return true;
// }

// =====================================================
// 初始化 ONNX Session
// =====================================================
bool RunModule::init_onnx_session()
{
    try {
        session_ = std::make_unique<Ort::Session>(
            env_, onnx_model_path_.c_str(), session_options_);
    }
    catch (const std::exception& e) {
        RCLCPP_ERROR(node_->get_logger(), "[RunModule] Failed to create ONNX session: %s", e.what());
        return false;
    }
    return true;
}

// =====================================================
// tick()
// =====================================================
void RunModule::tick(const interface_protocol::msg::JointState& state)
{

    // RCLCPP_WARN(node_->get_logger(),
    // "Tick(): pos=%zu vel=%zu expected=%d",
    // state.position.size(), state.velocity.size(), joint_num_);

    if (!model_loaded_) return;

    if ((int)state.position.size() < joint_num_ ||
        (int)state.velocity.size() < joint_num_) {
        RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 2000,
                             "waiting for full joint_state...");
        return;
    }

    phase_time_acc_ += dt_;
    double phase = std::fmod(phase_time_acc_ / cycle_time_, 1.0);

    Float clock_sin = (Float)std::sin(phase * 2.0 * M_PI);
    Float clock_cos = (Float)std::cos(phase * 2.0 * M_PI);

    FrameTensor current_frame;
    current_frame.data.reserve(frame_dim_);
    build_single_frame(state, clock_sin, clock_cos, current_frame);

    update_history(current_frame);

    std::vector<Float> obs;
    obs.reserve(obs_dim_);
    build_stacked_obs(obs);

    std::vector<Float> action_raw(action_dim_, 0.0f);
    if (!run_inference(obs, action_raw)) {
        RCLCPP_ERROR_THROTTLE(node_->get_logger(), *node_->get_clock(), 2000, "inference failed");
        return;
    }

    postprocess_and_publish(action_raw, state);
}

// =====================================================
// 单帧观测（已加 joint mapping）
// =====================================================
void RunModule::build_single_frame(
    const interface_protocol::msg::JointState& state,
    Float clock_sin,
    Float clock_cos,
    FrameTensor& frame)
{
    frame.data.clear();
    frame.data.reserve(frame_dim_);

    // ang_vel (0,0,0)
    for (int i = 0; i < 3; ++i) frame.data.push_back(0.0f);

    // projected gravity (0,0,-1)
    frame.data.push_back(0.0f);
    frame.data.push_back(0.0f);
    frame.data.push_back(-1.0f);

    // cmd (zero)
    frame.data.push_back(0.0f);
    frame.data.push_back(0.0f);
    frame.data.push_back(0.0f);

    // clock
    frame.data.push_back(clock_sin);
    frame.data.push_back(clock_cos);

    // ========== q with mapping ==========
    for (int i = 0; i < joint_num_; ++i) {
        int hw = kPolicyToHw[i];
        Float v = (Float)state.position[hw] * scale_dof_pos_;
        frame.data.push_back(clamp(v, -obs_clip_, obs_clip_));
    }

    // ========== dq with mapping ==========
    for (int i = 0; i < joint_num_; ++i) {
        int hw = kPolicyToHw[i];
        Float v = (Float)state.velocity[hw] * scale_dof_vel_;
        frame.data.push_back(clamp(v, -obs_clip_, obs_clip_));
    }

    // last action (policy order)
    for (int i = 0; i < joint_num_; ++i) {
        Float v = last_action_[i] * scale_last_action_;
        frame.data.push_back(clamp(v, -obs_clip_, obs_clip_));
    }
}

// =====================================================
void RunModule::update_history(const FrameTensor& frame)
{
    history_.push(frame);
}

void RunModule::build_stacked_obs(std::vector<Float>& obs)
{
    obs.clear();
    obs.reserve(obs_dim_);

    for (size_t i = 0; i < history_.size(); i++) {
        const auto& f = history_.at(i);
        obs.insert(obs.end(), f.data.begin(), f.data.end());
    }
}

// =====================================================
// 推理
// =====================================================
bool RunModule::run_inference(const std::vector<Float>& obs,
                              std::vector<Float>& action_raw)
{
    if (!session_) return false;

    std::array<int64_t, 2> input_shape{1, (int64_t)obs_dim_};

    Ort::Value input_tensor = Ort::Value::CreateTensor<Float>(
        memory_info_,
        const_cast<Float*>(obs.data()),
        obs.size(),
        input_shape.data(),
        input_shape.size());

    const char* input_names[]  = { input_node_name_.c_str() };
    const char* output_names[] = { output_node_name_.c_str() };

    auto outputs = session_->Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1,
        output_names, 1);

    Float* out_ptr = outputs[0].GetTensorMutableData<Float>();
    for (size_t i = 0; i < action_dim_; ++i)
        action_raw[i] = out_ptr[i];

    return true;
}

// =====================================================
// PostProcess + 发布（已加 mapping）
// =====================================================
void RunModule::postprocess_and_publish(
    const std::vector<Float>& action_raw,
    const interface_protocol::msg::JointState& state)
{
    std::vector<Float> clipped(action_dim_), scaled(action_dim_), final(action_dim_);

    // 1. clip
    for (size_t i = 0; i < action_dim_; i++)
        clipped[i] = clamp(action_raw[i], -act_clip_, act_clip_);

    last_action_ = clipped;

    // 2. scale + default pos
    for (int i = 0; i < joint_num_; i++)
        scaled[i] = clipped[i] * action_scale_[i] + default_position_[i];

    // 3. joint clip
    for (int i = 0; i < joint_num_; i++)
        final[i] = clamp(scaled[i], joint_clip_lower_[i], joint_clip_upper_[i]);

    // 4. publish（按 mapping 写到硬件 index）
    interface_protocol::msg::JointCommand cmd;
    cmd.position.resize(joint_num_);
    cmd.velocity.resize(joint_num_, 0.0);
    cmd.torque.resize(joint_num_, 0.0);
    cmd.feed_forward_torque.resize(joint_num_, 0.0);
    cmd.stiffness.resize(joint_num_);
    cmd.damping.resize(joint_num_);
    cmd.parallel_parser_type = 0;

    for (int i = 0; i < joint_num_; i++) {
        int hw = kPolicyToHw[i];        // 🔥关键：映射
        cmd.position[hw]  = final[i];
        cmd.stiffness[hw] = run_kp_[i];
        cmd.damping[hw]   = run_kd_[i];
    }

    pub_->publish(cmd);
}
