#pragma once

#include <rclcpp/rclcpp.hpp>
#include "interface_protocol/msg/joint_state.hpp"
#include "interface_protocol/msg/joint_command.hpp"

#include <onnxruntime_cxx_api.h>

#include <vector>
#include <string>
#include <cmath>
#include <memory>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/// RunModule: 类似 z::HumanLabInferenceWorker 的结构，
/// 但适配到 ROS2 + JointState / JointCommand。
class RunModule
{
public:
    RunModule(
        rclcpp::Node* node,
        rclcpp::Publisher<interface_protocol::msg::JointCommand>::SharedPtr pub,
        int joint_num);

    /// 每个控制周期在 FSMController::RUN 中调用
    void tick(const interface_protocol::msg::JointState& state);

    /// 是否成功加载模型 & 参数
    bool isReady() const { return model_loaded_; }

private:
    using Float = float;

    /// 一帧观测（已经 scale + clip 后）
    struct FrameTensor
    {
        std::vector<Float> data;
    };

    /// 简单 RingBuffer<T>，固定容量，覆盖旧数据
    template <typename T>
    class RingBuffer
    {
    public:
        RingBuffer() = default;

        void init(size_t capacity, const T& init_val)
        {
            capacity_ = capacity;
            buffer_.assign(capacity_, init_val);
            size_ = capacity_;     // 一开始用 init_val 填满
            head_ = capacity_ ? (capacity_ - 1) : 0;
        }

        void push(const T& v)
        {
            if (capacity_ == 0) return;
            head_ = (head_ + 1) % capacity_;
            buffer_[head_] = v;
            if (size_ < capacity_) {
                size_++;
            }
        }

        /// idx=0 表示最老的那一帧
        const T& at(size_t idx) const
        {
            if (size_ == 0) return buffer_[0];
            if (size_ < capacity_) {
                // 尚未推满，按顺序放在 [0, size_)
                return buffer_[idx];
            }
            size_t start = (head_ + 1) % capacity_;
            size_t i = (start + idx) % capacity_;
            return buffer_[i];
        }

        size_t size() const { return size_; }
        size_t capacity() const { return capacity_; }

    private:
        std::vector<T> buffer_;
        size_t capacity_{0};
        size_t size_{0};
        size_t head_{0};
    };

private:
    // ===== 初始化相关 =====
    bool load_params_from_ros();
    bool init_onnx_session();

    // ===== 构造观测 =====
    void build_single_frame(
        const interface_protocol::msg::JointState& state,
        Float clock_sin,
        Float clock_cos,
        FrameTensor& frame);

    void update_history(const FrameTensor& frame);

    void build_stacked_obs(std::vector<Float>& obs);

    // ===== ONNX 推理 =====
    bool run_inference(const std::vector<Float>& obs,
                       std::vector<Float>& action_raw);

    // ===== 后处理 + 发布 =====
    void postprocess_and_publish(
        const std::vector<Float>& action_raw,
        const interface_protocol::msg::JointState& state);

    // ===== 工具函数 =====
    template <typename T>
    static T clamp(T v, T lo, T hi)
    {
        return std::max(lo, std::min(hi, v));
    }

private:
    // ========== 基本 ROS 对象 ==========
    rclcpp::Node* node_;
    rclcpp::Publisher<interface_protocol::msg::JointCommand>::SharedPtr pub_;
    int joint_num_ = 12;

    // ========== ONNX Runtime 相关 ==========
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_;

    bool model_loaded_{false};

    std::string onnx_model_path_;
    std::string input_node_name_;
    std::string output_node_name_;

    // ========== 结构参数 ==========
    int    input_stack_len_{1};      // 对应 INPUT_STUCK_LENGTH
    double cycle_time_{0.6};          // 步态周期
    double dt_{0.02};                // 控制周期（和 FSM 定时器一致）

    size_t frame_dim_{0};             // 单帧维度
    size_t obs_dim_{0};               // 堆叠后总维度
    size_t action_dim_{0};            // 动作维度（通常 == joint_num_）

    // ========== 相位 / clock ==========
    double phase_time_acc_{0.0};      // 累积时间，用于 sin/cos

    // ========== 输入堆叠 ==========
    RingBuffer<FrameTensor> history_;

    // ========== RL 参数（来自 YAML） ==========
    // obs 裁剪
    Float obs_clip_{100.0f};
    // action 裁剪
    Float act_clip_{1.0f};

    // 观测 scale（对应 CommonLocoInferenceWorker）
    Float scale_lin_vel_{1.0f};
    Float scale_ang_vel_{1.0f};
    Float scale_project_gravity_{1.0f};
    Float scale_cmd_{1.0f};
    Float scale_dof_pos_{1.0f};
    Float scale_dof_vel_{1.0f};
    Float scale_last_action_{1.0f};

    // 动作 scale、默认姿态和关节裁剪
    std::vector<Float> action_scale_;      // [joint_num_]
    std::vector<Float> default_position_;  // [joint_num_]
    std::vector<Float> joint_clip_upper_;  // [joint_num_]
    std::vector<Float> joint_clip_lower_;  // [joint_num_]

    // 运行时：上一时刻动作（未 scale 的）
    std::vector<Float> last_action_;       // [joint_num_]

    // 运行时：关节 PD（直接从 ROS 参数取）
    std::vector<double> run_kp_;           // [joint_num_]
    std::vector<double> run_kd_;           // [joint_num_]
};
