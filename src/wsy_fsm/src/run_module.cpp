#include "wsy_fsm/run_module.hpp"

#include <chrono>
#include <iostream>

RunModule::RunModule(
    rclcpp::Node* node,
    rclcpp::Publisher<interface_protocol::msg::JointCommand>::SharedPtr pub,
    int joint_num)
: node_(node)
, pub_(pub)
, joint_num_(12)
, env_(ORT_LOGGING_LEVEL_WARNING, "fsm_run")
, session_options_{}
, memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    // 图优化 + 线程设置（类似 AbstractNetInferenceWorker）
    session_options_.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetExecutionMode(ORT_SEQUENTIAL);

    action_dim_ = static_cast<size_t>(joint_num_);

    // 从 ROS 参数加载所有配置（YAML）
    if (!load_params_from_ros()) {
        RCLCPP_ERROR(node_->get_logger(), "[RunModule] load_params_from_ros failed.");
        model_loaded_ = false;
        return;
    }

    // 初始化历史 Buffer
    FrameTensor zero_frame;
    zero_frame.data.assign(frame_dim_, 0.0f);
    history_.init(static_cast<size_t>(input_stack_len_), zero_frame);

    // last_action_ 初始为 0
    last_action_.assign(joint_num_, 0.0f);

    // 初始化 ONNX Session
    if (!init_onnx_session()) {
        model_loaded_ = false;
        return;
    }

    model_loaded_ = true;
    RCLCPP_INFO(node_->get_logger(),
                "[RunModule] Init OK. model=%s, input_stack=%d, frame_dim=%zu, obs_dim=%zu, action_dim=%zu",
                onnx_model_path_.c_str(), input_stack_len_, frame_dim_, obs_dim_, action_dim_);
}

// =====================================================
// 从 ROS 参数(YAML)加载配置
// =====================================================
bool RunModule::load_params_from_ros()
{
    // ==== 基本推理参数 ====
    onnx_model_path_  = node_->declare_parameter<std::string>("onnx_model_path", "/home/wsy/rl/CtrlZ_sim/Simulation/GB2/main_stand/checkpoints/policy.onnx");
    input_stack_len_  = node_->declare_parameter<int>("input_stack_len", 10);
    cycle_time_       = node_->declare_parameter<double>("cycle_time", 0.8);
    dt_               = node_->declare_parameter<double>("control_dt", 0.01);

    input_node_name_  = node_->declare_parameter<std::string>("input_node_name", "obs");
    output_node_name_ = node_->declare_parameter<std::string>("output_node_name", "actions");

    if (input_stack_len_ <= 0) input_stack_len_ = 1;
    if (cycle_time_ <= 0.0)    cycle_time_ = 1.0;   // 防止除 0
    if (dt_ <= 0.0)            dt_ = 0.002;

    // ==== 观测裁剪 & 动作裁剪 ====
    obs_clip_ = static_cast<Float>(node_->declare_parameter<double>("obs_clip", 100.0));
    act_clip_ = static_cast<Float>(node_->declare_parameter<double>("act_clip", 1.0));

    // ==== 观测 scale（对应 CommonLocoInferenceWorker::Scales_*） ====
    // scale_lin_vel_         = static_cast<Float>(node_->declare_parameter<double>("scale_lin_vel", 1.0));
    scale_ang_vel_         = static_cast<Float>(node_->declare_parameter<double>("scale_ang_vel", 1.0));
    scale_project_gravity_ = static_cast<Float>(node_->declare_parameter<double>("scale_project_gravity", 1.0));
    scale_cmd_             = static_cast<Float>(node_->declare_parameter<double>("scale_cmd", 1.0));
    scale_dof_pos_         = static_cast<Float>(node_->declare_parameter<double>("scale_dof_pos", 1.0));
    scale_dof_vel_         = static_cast<Float>(node_->declare_parameter<double>("scale_dof_vel", 1.0));
    scale_last_action_     = static_cast<Float>(node_->declare_parameter<double>("scale_last_action", 1.0));

    // ==== 动作 scale / clip / joint limits / default pos ====
    auto vec_default = std::vector<double>(joint_num_, 0.0);
    auto vec_one     = std::vector<double>(joint_num_, 1.0);

    // action_scale
    auto action_scale_d = node_->declare_parameter<std::vector<double>>("action_scale", vec_one);
    // default_position
    auto default_pos_d  = node_->declare_parameter<std::vector<double>>("default_position", vec_default);
    // joint_clip_upper / lower
    auto joint_upper_d  = node_->declare_parameter<std::vector<double>>("joint_clip_upper", vec_one);
    auto joint_lower_d  = node_->declare_parameter<std::vector<double>>("joint_clip_lower", vec_default);

    if ((int)action_scale_d.size()  != joint_num_ ||
        (int)default_pos_d.size()   != joint_num_ ||
        (int)joint_upper_d.size()   != joint_num_ ||
        (int)joint_lower_d.size()   != joint_num_) {
        RCLCPP_ERROR(node_->get_logger(),
            "[RunModule] YAML vector size mismatch: expected %d, action_scale=%zu, default_pos=%zu, upper=%zu, lower=%zu",
            joint_num_, action_scale_d.size(), default_pos_d.size(), joint_upper_d.size(), joint_lower_d.size());
        return false;
    }

    action_scale_.resize(joint_num_);
    default_position_.resize(joint_num_);
    joint_clip_upper_.resize(joint_num_);
    joint_clip_lower_.resize(joint_num_);

    for (int i = 0; i < joint_num_; ++i) {
        action_scale_[i]     = static_cast<Float>(action_scale_d[i]);
        default_position_[i] = static_cast<Float>(default_pos_d[i]);
        joint_clip_upper_[i] = static_cast<Float>(joint_upper_d[i]);
        joint_clip_lower_[i] = static_cast<Float>(joint_lower_d[i]);
    }

    // ==== 运行时 PD（Run 状态下下发的 Kp / Kd） ====
    run_kp_ = node_->declare_parameter<std::vector<double>>("run_kp", std::vector<double>(joint_num_, 40.0));
    run_kd_ = node_->declare_parameter<std::vector<double>>("run_kd", std::vector<double>(joint_num_, 1.0));

    if ((int)run_kp_.size() != joint_num_ || (int)run_kd_.size() != joint_num_) {
        RCLCPP_ERROR(node_->get_logger(),
            "[RunModule] run_kp/run_kd size mismatch: expected=%d, got Kp=%zu, Kd=%zu",
            joint_num_, run_kp_.size(), run_kd_.size());
        return false;
    }

    // ==== 计算单帧维度 & 总观测维度 ====
    // frame: [ lin_vel(3) + ang_vel(3) + proj_grav(3) + cmd(3) + clock(2)
    //        + q(N) + dq(N) + last_action(N) ]
    frame_dim_ =                  // lin_vel
                 3                 // ang_vel
               + 3                 // project_gravity
               + 3                 // cmd
               + 2                 // clock
               + joint_num_        // q
               + joint_num_        // dq
               + joint_num_;       // last_action

    obs_dim_ = frame_dim_ * static_cast<size_t>(input_stack_len_);

    return true;
}

// =====================================================
// 初始化 ONNX Session
// =====================================================
bool RunModule::init_onnx_session()
{
    try {
        session_ = std::make_unique<Ort::Session>(env_, onnx_model_path_.c_str(), session_options_);
    }
    catch (const std::exception& e) {
        RCLCPP_ERROR(node_->get_logger(),
            "[RunModule] Failed to create ONNX session: %s", e.what());
        return false;
    }
    return true;
}

// =====================================================
// tick：每个 RUN 控制周期调用
// =====================================================
void RunModule::tick(const interface_protocol::msg::JointState& state)
{
    if (!model_loaded_) {
        RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 3000,
                             "[RunModule] tick() but model not loaded.");
        return;
    }
    if ((int)state.position.size() < joint_num_ ||
        (int)state.velocity.size() < joint_num_) {
        RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 2000,
                             "[RunModule] waiting for full joint_state.");
        return;
    }

    // 1) 更新相位 clock
    phase_time_acc_ += dt_;
    double phase = std::fmod(phase_time_acc_ / cycle_time_, 1.0);  // [0,1)
    Float clock_sin = static_cast<Float>(std::sin(phase * 2.0 * M_PI));
    Float clock_cos = static_cast<Float>(std::cos(phase * 2.0 * M_PI));

    // 2) 构造当前帧 + scale/clip
    FrameTensor current_frame;
    current_frame.data.reserve(frame_dim_);
    build_single_frame(state, clock_sin, clock_cos, current_frame);

    // 3) 更新历史堆叠
    update_history(current_frame);

    // 4) 构造最终 obs
    std::vector<Float> obs;
    obs.reserve(obs_dim_);
    build_stacked_obs(obs);

    // 5) ONNX 推理
    std::vector<Float> action_raw(action_dim_, 0.0f);
    if (!run_inference(obs, action_raw)) {
        RCLCPP_ERROR_THROTTLE(node_->get_logger(), *node_->get_clock(), 2000,
                              "[RunModule] inference failed.");
        return;
    }

    // 6) PostProcess + 发布
    postprocess_and_publish(action_raw, state);
}

// =====================================================
// 单帧特征构造（类似 HumanLabInferenceWorker::SingleInputVecScaled）
// =====================================================
void RunModule::build_single_frame(
    const interface_protocol::msg::JointState& state,
    Float clock_sin,
    Float clock_cos,
    FrameTensor& frame)
{
    frame.data.clear();
    frame.data.reserve(frame_dim_);

    // 先用 0 填充线速度和角速度，之后你可以替换为 IMU / 估计值
    // lin_vel (3)
    // for (int i = 0; i < 3; ++i) {
    //     Float v = 0.0f * scale_lin_vel_;
    //     v = clamp(v, -obs_clip_, obs_clip_);
    //     frame.data.push_back(v);
    // }

    // ang_vel (3)
    for (int i = 0; i < 3; ++i) {
        Float v = 0.0f * scale_ang_vel_;
        v = clamp(v, -obs_clip_, obs_clip_);
        frame.data.push_back(v);
    }

    // projected gravity (3) 先用 (0,0,-1)
    {
        Float gx = 0.0f  * scale_project_gravity_;
        Float gy = 0.0f  * scale_project_gravity_;
        Float gz = -1.0f * scale_project_gravity_;
        frame.data.push_back(clamp(gx, -obs_clip_, obs_clip_));
        frame.data.push_back(clamp(gy, -obs_clip_, obs_clip_));
        frame.data.push_back(clamp(gz, -obs_clip_, obs_clip_));
    }

    // user command (3) 先全 0
    for (int i = 0; i < 3; ++i) {
        Float v = 0.0f * scale_cmd_;
        v = clamp(v, -obs_clip_, obs_clip_);
        frame.data.push_back(v);
    }

    // clock (2)
    frame.data.push_back(clamp(clock_sin, -obs_clip_, obs_clip_));
    frame.data.push_back(clamp(clock_cos, -obs_clip_, obs_clip_));

    // q
    for (int i = 0; i < joint_num_; ++i) {
        Float v = static_cast<Float>(state.position[i]) * scale_dof_pos_;
        frame.data.push_back(clamp(v, -obs_clip_, obs_clip_));
    }

    // dq
    for (int i = 0; i < joint_num_; ++i) {
        Float v = static_cast<Float>(state.velocity[i]) * scale_dof_vel_;
        frame.data.push_back(clamp(v, -obs_clip_, obs_clip_));
    }

    // last_action
    for (int i = 0; i < joint_num_; ++i) {
        Float v = last_action_[i] * scale_last_action_;
        frame.data.push_back(clamp(v, -obs_clip_, obs_clip_));
    }

    if (frame.data.size() != frame_dim_) {
        RCLCPP_ERROR(node_->get_logger(),
            "[RunModule] build_single_frame size mismatch: got=%zu, expected=%zu",
            frame.data.size(), frame_dim_);
    }
}

// =====================================================
// 更新 RingBuffer
// =====================================================
void RunModule::update_history(const FrameTensor& frame)
{
    history_.push(frame);
}

// =====================================================
// 将 RingBuffer 展开为 obs 向量
// =====================================================
void RunModule::build_stacked_obs(std::vector<Float>& obs)
{
    obs.clear();
    obs.reserve(obs_dim_);

    size_t n = history_.size();
    for (size_t i = 0; i < n; ++i) {
        const auto& f = history_.at(i);
        obs.insert(obs.end(), f.data.begin(), f.data.end());
    }

    if (obs.size() != obs_dim_) {
        RCLCPP_ERROR(node_->get_logger(),
            "[RunModule] build_stacked_obs size mismatch: got=%zu, expected=%zu",
            obs.size(), obs_dim_);
    }
}

// =====================================================
// ONNX 推理（简单版）
// =====================================================
bool RunModule::run_inference(const std::vector<Float>& obs,
                              std::vector<Float>& action_raw)
{
    if (!session_) {
        return false;
    }

    // 构造 input tensor
    std::array<int64_t, 2> input_shape{1, static_cast<int64_t>(obs_dim_)};

    Ort::Value input_tensor = Ort::Value::CreateTensor<Float>(
        memory_info_,
        const_cast<Float*>(obs.data()),   // ORT 需要非 const 指针
        obs.size(),
        input_shape.data(),
        input_shape.size()
    );

    const char* input_names[]  = { input_node_name_.c_str() };
    const char* output_names[] = { output_node_name_.c_str() };

    std::vector<Ort::Value> outputs = session_->Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,
        output_names,
        1
    );

    if (outputs.empty()) {
        RCLCPP_ERROR(node_->get_logger(), "[RunModule] ONNX returned no outputs.");
        return false;
    }

    Float* out_ptr = outputs[0].GetTensorMutableData<Float>();

    // 这里假设输出长度 == action_dim_
    for (size_t i = 0; i < action_dim_; ++i) {
        action_raw[i] = out_ptr[i];
    }

    return true;
}

// =====================================================
// PostProcess + 发布 JointCommand
// 对应 HumanLabInferenceWorker::PostProcess：
//  1) clip action → last_action
//  2) scale + default_position
//  3) joint clip
// =====================================================
void RunModule::postprocess_and_publish(
    const std::vector<Float>& action_raw,
    const interface_protocol::msg::JointState& state)
{
    std::vector<Float> clipped_action(action_dim_, 0.0f);
    std::vector<Float> scaled_action(action_dim_, 0.0f);
    std::vector<Float> final_action(action_dim_, 0.0f);

    // 1) clip 动作，记为 last_action_
    for (size_t i = 0; i < action_dim_; ++i) {
        Float a = clamp(action_raw[i], -act_clip_, act_clip_);
        clipped_action[i] = a;
    }
    last_action_ = clipped_action;

    // 2) scale + default_position
    for (int i = 0; i < joint_num_; ++i) {
        Float a = clipped_action[i] * action_scale_[i] + default_position_[i];
        scaled_action[i] = a;
    }

    // 3) joint clip
    for (int i = 0; i < joint_num_; ++i) {
        Float a = clamp(scaled_action[i], joint_clip_lower_[i], joint_clip_upper_[i]);
        final_action[i] = a;
    }

    // 4) 发布 JointCommand
    interface_protocol::msg::JointCommand cmd;
    cmd.position.resize(joint_num_);
    cmd.velocity.resize(joint_num_, 0.0);
    cmd.torque.resize(joint_num_, 0.0);
    cmd.feed_forward_torque.resize(joint_num_, 0.0);
    cmd.stiffness.resize(joint_num_);
    cmd.damping.resize(joint_num_);
    cmd.parallel_parser_type = 0;

    for (int i = 0; i < joint_num_; ++i) {
        cmd.position[i] = static_cast<double>(final_action[i]);
        cmd.stiffness[i] = run_kp_[i];
        cmd.damping[i]   = run_kd_[i];
    }

    pub_->publish(cmd);
}
