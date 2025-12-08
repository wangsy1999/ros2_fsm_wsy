#pragma once

#include <onnxruntime_cxx_api.h>
#include <rclcpp/rclcpp.hpp>

#include <memory>
#include <string>
#include <vector>

/**
 * @brief Thin wrapper around ONNX Runtime to load a policy model and run a single forward pass.
 */
class OnnxRuntimeRunner
{
public:
    struct Config
    {
        std::string model_path;
        size_t input_dim{0};
        size_t output_dim{0};
        int intra_op_num_threads{1};
        bool use_cpu{true};
    };

    explicit OnnxRuntimeRunner(rclcpp::Logger logger);

    bool load(const Config& cfg);
    bool ready() const { return session_ != nullptr; }
    bool infer(const std::vector<float>& obs, std::vector<float>& action_out);
    const Config& config() const { return cfg_; }

private:
    rclcpp::Logger logger_;
    Config cfg_;

    Ort::Env env_;
    Ort::SessionOptions session_opts_;
    Ort::MemoryInfo mem_info_;
    std::unique_ptr<Ort::Session> session_;

    std::string input_name_;
    std::string output_name_;
    std::vector<int64_t> input_shape_;
    std::vector<int64_t> output_shape_;
};
