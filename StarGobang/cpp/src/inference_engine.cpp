/*
 * MIT License
 * 
 * Copyright (c) 2026 StarGobang Team
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "gobang_engine.h"
#include "mcts.h"
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <random>

namespace gobang {

GobangEngine::GobangEngine(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "GobangEngine"),
      mcts_config_{},
      model_loaded_(false),
      meta_learner_(std::make_unique<MetaLearner>()) {
    
    // Initialize HPC scheduler
    initialize_hpc();
    
    // Initialize ONNX Runtime session options
    Ort::SessionOptions session_options;
    
    // Configure ONNX threads via meta learner
    const auto& params = meta_learner_->get_current_params();
    session_options.SetIntraOpNumThreads(params.onnx_intra_threads);
    session_options.SetInterOpNumThreads(params.onnx_inter_threads);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    
    // Load neural network model
    // Model source: model_gobang.onnx exported from Python train_loop.py closed-loop training
    try {
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);
        model_loaded_ = true;
        std::cout << "[INFO] Pure inference engine loaded: " << model_path << std::endl;
    } catch (const Ort::Exception& e) {
        // Fallback mode on model load failure
        std::cerr << "Warning: Failed to load ONNX model (" << e.what() 
                  << "). Using fallback mode." << std::endl;
        model_loaded_ = false;
    }
    
    // Set input/output node names
    input_names_.push_back("input");
    output_names_.push_back("policy");
    output_names_.push_back("value");
    
    // Optimize ONNX session
    optimize_onnx_session();
}

GobangEngine::~GobangEngine() = default;

void GobangEngine::set_mcts_config(const MCTSConfig& config) {
    mcts_config_ = config;
}

MetaLearner& GobangEngine::get_meta_learner() {
    return *meta_learner_;
}

HPCScheduler& GobangEngine::get_hpc_scheduler() {
    return HPCScheduler::instance();
}

void GobangEngine::set_performance_mode(MetaLearningConfig::PerformanceMode mode) {
    switch (mode) {
        case MetaLearningConfig::PerformanceMode::MAX_STRENGTH:
            meta_learner_->set_state(MetaLearningState::AGGRESSIVE_TUNING);
            break;
        case MetaLearningConfig::PerformanceMode::BALANCED:
            meta_learner_->set_state(MetaLearningState::ONLINE_ADAPTATION);
            break;
        case MetaLearningConfig::PerformanceMode::FAST:
            meta_learner_->set_state(MetaLearningState::CONSERVATIVE_STABLE);
            break;
    }
}

MetaLearningConfig::PerformanceMode GobangEngine::get_performance_mode() const {
    auto state = meta_learner_->get_state();
    switch (state) {
        case MetaLearningState::AGGRESSIVE_TUNING:
            return MetaLearningConfig::PerformanceMode::MAX_STRENGTH;
        case MetaLearningState::ONLINE_ADAPTATION:
            return MetaLearningConfig::PerformanceMode::BALANCED;
        case MetaLearningState::CONSERVATIVE_STABLE:
        case MetaLearningState::INFERENCE_ONLY:
            return MetaLearningConfig::PerformanceMode::FAST;
        default:
            return MetaLearningConfig::PerformanceMode::FAST;
    }
}

void GobangEngine::online_adapt_step() {
    // Online fine-tuning via meta learning and learning monitor
}

std::pair<std::vector<float>, float> GobangEngine::run_inference(const Board& board) {
    HPC_PROFILE("NN_Inference");
    
    // Return random policy if model not loaded
    if (!model_loaded_ || !session_) {
        std::vector<float> random_policy(NetworkSpec::POLICY_OUTPUT, 1.0f / NetworkSpec::POLICY_OUTPUT);
        return {random_policy, 0.0f};
    }
    
    // Get current optimal parameters
    const auto& params = meta_learner_->get_current_params();
    
    // Create input feature tensor
    auto features = board.create_feature_tensor();
    
    // Input tensor shape: [batch_size=1, channels=10, height=15, width=15]
    std::array<int64_t, 4> input_shape{
        static_cast<int64_t>(NetworkSpec::BATCH_SIZE),
        static_cast<int64_t>(NetworkSpec::INPUT_CHANNELS),
        static_cast<int64_t>(NetworkSpec::HEIGHT),
        static_cast<int64_t>(NetworkSpec::WIDTH)
    };
    
    // Create input tensor
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    size_t input_tensor_size = NetworkSpec::BATCH_SIZE * NetworkSpec::INPUT_CHANNELS * 
                               NetworkSpec::HEIGHT * NetworkSpec::WIDTH;
    
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, 
        features.data(), 
        input_tensor_size,
        input_shape.data(), 
        input_shape.size()
    );
    
    // Execute ONNX Runtime inference
    std::vector<Ort::Value> output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_names_.data(),
        &input_tensor,
        input_names_.size(),
        output_names_.data(),
        output_names_.size()
    );
    
    // Extract policy output
    float* policy_data = output_tensors[0].GetTensorMutableData<float>();
    std::vector<float> policy(policy_data, policy_data + NetworkSpec::POLICY_OUTPUT);
    
    // Extract value output
    float* value_data = output_tensors[1].GetTensorMutableData<float>();
    float value = value_data[0];
    
    // Softmax 归一化 policy
    float max_val = *std::max_element(policy.begin(), policy.end());
    float sum = 0.0f;
    for (float& p : policy) {
        p = std::exp(p - max_val);
        sum += p;
    }
    for (float& p : policy) {
        p /= sum;
    }
    
    // 记录性能指标
    PerformanceMetrics metrics;
    metrics.inference_time_ms = 10.0f;  // 简化，实际应测量
    meta_learner_->online_feedback(metrics);
    
    return {policy, value};
}

std::pair<int, int> GobangEngine::get_best_move(const Board& board) {
    HPC_PROFILE("GetBestMove");
    
    // 使用 MCTS 搜索最佳移动
    if (!mcts_) {
        mcts_ = std::make_unique<MCTS>(this, meta_learner_.get());
    }
    
    // 获取当前最优参数
    const auto& params = meta_learner_->get_current_params();
    
    // 根据对局阶段动态调整
    meta_learner_->adapt_to_game_stage(
        board.move_count(),
        0.5f,  // 简化胜率估计，实际应从 value 得到
        0.0f   // 无时间限制
    );
    
    // 使用并行搜索（如果硬件支持）
    if (params.mcts_parallel_threads > 1) {
        return mcts_->parallel_search(board, params.mcts_parallel_threads);
    } else {
        return mcts_->search(board);
    }
}

void GobangEngine::initialize_hpc() {
    // 初始化 HPC 调度器
    auto& hpc = HPCScheduler::instance();
    hpc.initialize(meta_learner_->get_hardware());
}

void GobangEngine::optimize_onnx_session() {
    // 进一步优化 ONNX 会话
    // 可以使用内存优化、执行优化等
    // 简化实现
}

} // namespace gobang
