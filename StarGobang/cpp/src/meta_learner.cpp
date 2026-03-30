// 元学习核心实现 - 统一接管硬件感知、自适应调参与在线微调
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
#include "meta_learner.h"
#include <cstring>
#include <fstream>
#include <cmath>
#include <algorithm>

#ifdef __linux__
#include <sys/sysinfo.h>
#include <unistd.h>
#include <sched.h>
#endif

namespace gobang {

// ========== HardwareResources 实现 ==========

HardwareResources::HardwareResources()
    : cpu_cores(4), available_threads(4), total_memory_mb(8192), available_memory_mb(4096),
      has_gpu(false), gpu_memory_mb(0), available_gpu_memory_mb(0), gpu_compute_score(0.0f),
      l1_cache_size(32768), l2_cache_size(262144), l3_cache_size(8388608), numa_nodes(1) {
    
    // 检测硬件资源
    #ifdef __linux__
    cpu_cores = static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));
    available_threads = cpu_cores;
    
    // 读取内存信息
    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        total_memory_mb = si.totalram * si.mem_unit / (1024 * 1024);
        available_memory_mb = si.freeram * si.mem_unit / (1024 * 1024);
    }
    
    // 简单的缓存大小估计
    l1_cache_size = 32 * 1024;   // 32KB
    l2_cache_size = 256 * 1024;  // 256KB
    l3_cache_size = 8 * 1024 * 1024;  // 8MB
    
    // NUMA 节点数
    numa_nodes = 1;  // 简化，实际可读取/sys/devices/system/node/
    #endif
}

// ========== MetaParams 实现 ==========

MetaParams::MetaParams()
    : mcts_simulations(800), mcts_c_puct(1.5f), mcts_dirichlet_alpha(0.3f),
      mcts_noise_scale(0.25f), mcts_virtual_loss(3), mcts_parallel_threads(1),
      use_fp16(false), onnx_intra_threads(1), onnx_inter_threads(1), use_cpu_affinity(false),
      batch_size(1), enable_tensor_core(false), transposition_table_size(256),
      enable_cache_prefetch(true), feature_planes(10), enable_dynamic_features(true),
      decision_temperature(1.0f), temperature_decay(0.99f), enable_online_adaptation(true),
      online_lr(1e-4f), online_update_freq(10), online_batch_size(8),
      enable_memory_pool(true), memory_pool_size_mb(512) {}

void MetaParams::init_from_hardware(const HardwareResources& hw) {
    // 根据硬件资源自动设置最优参数
    
    // MCTS 并行度
    mcts_parallel_threads = std::max(1, hw.cpu_cores - 1);  // 保留 1 核给系统
    
    // ONNX 线程数
    onnx_intra_threads = std::min(4, hw.cpu_cores / 2);
    onnx_inter_threads = std::min(2, hw.cpu_cores / 4);
    
    // CPU 亲和性
    use_cpu_affinity = (hw.cpu_cores >= 4);
    
    // 内存池大小（使用可用内存的 50%）
    memory_pool_size_mb = std::min(static_cast<size_t>(512), 
                                    hw.available_memory_mb / 2);
    
    // 置换表大小
    transposition_table_size = std::min(1024, 
                                        static_cast<int>(hw.available_memory_mb / 8));
    
    // 根据 GPU 情况决定是否启用 FP16
    use_fp16 = hw.has_gpu && (hw.gpu_compute_score > 100.0f);
    enable_tensor_core = hw.has_gpu && (hw.gpu_memory_mb >= 8192);
    
    // 批处理大小
    batch_size = hw.has_gpu ? 4 : 1;
}

// ========== PerformanceMetrics 实现 ==========

PerformanceMetrics::PerformanceMetrics()
    : inference_time_ms(0.0f), positions_per_second(0.0f), search_depth(0.0f),
      nodes_per_second(0.0f), memory_used_mb(0), cache_hit_rate(0), elo_estimate(1500.0f),
      win_rate(0.5f), cpu_utilization(0.0f), gpu_utilization(0.0f), memory_bandwidth_util(0.0f) {}

// ========== MetaLearner 实现 ==========

MetaLearner::MetaLearner()
    : state_(MetaLearningState::INFERENCE_ONLY), max_history_size_(100),
      weights_initialized_(false) {}

MetaLearner::~MetaLearner() = default;

void MetaLearner::initialize(const HardwareResources& hw) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    hw_ = hw;
    current_params_.init_from_hardware(hw);
    
    // 初始化元学习权重（简化为启发式值，实际应由训练得到）
    // 这些权重决定了如何根据局面和硬件调整参数
    meta_weights_ = {
        1.0f,   // 模拟次数权重
        0.5f,   // PUCT 常数权重
        0.3f,   // Dirichlet alpha 权重
        0.2f,   // 噪声缩放权重
        0.4f,   // 虚拟损失权重
        0.8f,   // 并行线程权重
        0.6f,   // FP16 权重
        0.7f,   // ONNX 线程权重
        0.5f,   // 批处理权重
        0.9f,   // 缓存权重
        0.4f,   // 特征平面权重
        0.6f,   // 决策温度权重
        0.3f,   // 在线学习率权重
        0.5f,   // 内存池权重
        0.7f,   // CPU 亲和性权重
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f  // 预留
    };
    weights_initialized_ = true;
    
    // 应用 CPU 亲和性优化
    if (current_params_.use_cpu_affinity) {
        apply_cpu_affinity();
    }
    
    // 优化内存布局
    optimize_memory_layout();
}

const MetaParams& MetaLearner::get_current_params() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_params_;
}

void MetaLearner::adapt_to_game_stage(
    size_t move_count,
    float current_win_rate,
    float time_remaining_sec
) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 根据对局阶段动态调整参数
    
    // 开局阶段（前 10 步）：增加探索
    if (move_count < 10) {
        current_params_.mcts_c_puct = 2.0f;  // 更高的探索常数
        current_params_.mcts_dirichlet_alpha = 0.5f;
        current_params_.decision_temperature = 1.2f;  // 更随机的决策
    }
    // 中盘阶段（10-50 步）：平衡探索与利用
    else if (move_count < 50) {
        current_params_.mcts_c_puct = 1.5f;
        current_params_.decision_temperature = 1.0f;
    }
    // 残局阶段（50 步后）：减少探索，增加精确度
    else {
        current_params_.mcts_c_puct = 1.0f;
        current_params_.mcts_dirichlet_alpha = 0.1f;
        current_params_.decision_temperature = 0.8f;  // 更确定的决策
        
        // 如果有时间压力，减少模拟次数
        if (time_remaining_sec > 0.0f && time_remaining_sec < 30.0f) {
            emergency_optimize(30.0f / time_remaining_sec);
        }
    }
    
    // 根据胜率调整策略
    if (current_win_rate > 0.7f) {
        // 优势下稳健
        current_params_.mcts_virtual_loss = 5;
        current_params_.decision_temperature *= 0.9f;
    } else if (current_win_rate < 0.3f) {
        // 劣势下激进
        current_params_.mcts_virtual_loss = 1;
        current_params_.mcts_c_puct *= 1.2f;
    }
}

void MetaLearner::online_feedback(const PerformanceMetrics& metrics) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    metrics_ = metrics;
    history_.push_back(metrics);
    
    // 限制历史记录大小
    if (history_.size() > max_history_size_) {
        history_.erase(history_.begin());
    }
    
    // 基于性能反馈在线微调参数
    if (history_.size() >= 10) {
        // 分析性能趋势
        float recent_avg = 0.0f;
        float old_avg = 0.0f;
        
        for (size_t i = 0; i < 5; ++i) {
            recent_avg += history_[history_.size() - 1 - i].inference_time_ms;
            old_avg += history_[i].inference_time_ms;
        }
        recent_avg /= 5.0f;
        old_avg /= 5.0f;
        
        // 如果性能下降，调整参数
        if (recent_avg > old_avg * 1.1f) {
            // 推理时间增加，减少模拟次数或优化线程
            current_params_.mcts_simulations = static_cast<int>(
                current_params_.mcts_simulations * 0.9f
            );
        }
    }
}

const HardwareResources& MetaLearner::get_hardware() const {
    return hw_;
}

void MetaLearner::set_state(MetaLearningState state) {
    std::lock_guard<std::mutex> lock(mutex_);
    state_ = state;
    
    // 根据状态调整参数
    switch (state) {
        case MetaLearningState::INFERENCE_ONLY:
            current_params_.enable_online_adaptation = false;
            break;
            
        case MetaLearningState::ONLINE_ADAPTATION:
            current_params_.enable_online_adaptation = true;
            current_params_.online_lr = 1e-4f;
            break;
            
        case MetaLearningState::AGGRESSIVE_TUNING:
            current_params_.enable_online_adaptation = true;
            current_params_.online_lr = 1e-3f;  // 更高学习率
            current_params_.mcts_simulations = std::min(
                current_params_.mcts_simulations * 2,
                10000  // 使用固定最大值替代 max_simulations
            );
            break;
            
        case MetaLearningState::CONSERVATIVE_STABLE:
            current_params_.enable_online_adaptation = false;
            current_params_.mcts_virtual_loss = 5;  // 更保守
            break;
    }
}

MetaLearningState MetaLearner::get_state() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_;
}

const PerformanceMetrics& MetaLearner::get_metrics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return metrics_;
}

void MetaLearner::emergency_optimize(float time_pressure_factor) {
    // 时间压力下紧急优化
    float factor = 1.0f / time_pressure_factor;
    
    current_params_.mcts_simulations = std::max(
        100,  // 使用固定最小值替代 min_simulations
        static_cast<int>(current_params_.mcts_simulations * factor)
    );
    
    current_params_.mcts_parallel_threads = std::max(
        1,
        current_params_.mcts_parallel_threads - 1
    );
    
    current_params_.decision_temperature *= 0.8f;
}

void MetaLearner::reset_stats() {
    std::lock_guard<std::mutex> lock(mutex_);
    history_.clear();
    metrics_ = PerformanceMetrics{};
}

void MetaLearner::compute_optimal_params() {
    // 根据硬件和历史性能计算最优参数
    // 简化实现，实际应由元学习网络预测
    
    // 基于 Elo 估计调整模拟次数
    float elo_target = 3000.0f;  // 目标等级分
    float elo_ratio = std::min(1.0f, metrics_.elo_estimate / elo_target);
    
    current_params_.mcts_simulations = static_cast<int>(
        800 + (10000 - 800) * elo_ratio  // 使用固定最大值替代 max_simulations
    );
}

void MetaLearner::update_hardware_stats() {
    // 定期更新硬件统计信息
    // 简化实现
}

float MetaLearner::estimate_elo(const MetaParams& params) const {
    // 简化的 Elo 估计公式
    // 实际应通过大量对局统计得到
    
    float base_elo = 1500.0f;
    
    // 模拟次数贡献
    float sim_bonus = std::log2(params.mcts_simulations / 100.0f) * 100.0f;
    
    // 并行度贡献
    float parallel_bonus = params.mcts_parallel_threads * 20.0f;
    
    // 缓存贡献
    float cache_bonus = std::log2(params.transposition_table_size / 64.0f) * 50.0f;
    
    return base_elo + sim_bonus + parallel_bonus + cache_bonus;
}

void MetaLearner::apply_cpu_affinity() {
    #ifdef __linux__
    cpu_set_t mask;
    CPU_ZERO(&mask);
    
    // 绑定到前 N 个核心
    int num_cores = std::min(hw_.cpu_cores, current_params_.mcts_parallel_threads + 1);
    for (int i = 0; i < num_cores; ++i) {
        CPU_SET(i, &mask);
    }
    
    // 设置当前线程的亲和性
    sched_setaffinity(0, sizeof(mask), &mask);
    #endif
}

void MetaLearner::optimize_memory_layout() {
    // 优化内存布局以提高缓存命中率
    // 这主要通过数据结构的对齐和预取实现
    // 简化实现
}

} // namespace gobang
