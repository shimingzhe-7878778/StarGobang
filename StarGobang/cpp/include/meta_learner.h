// [元学习核心] 统一接管硬件感知、自适应调参与在线微调
// 核心设计原则：
// 1. 元学习在训练阶段学习如何学习，内置硬件感知能力
// 2. 推理阶段根据硬件资源自动动态调整所有影响棋力的参数
// 3. 不新增独立模块，所有逻辑由元学习统一接管
// 4. Ubuntu 22.04.5 极致性能优化，禁止 sudo 提权

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
#ifndef GOBANG_META_LEARNER_H
#define GOBANG_META_LEARNER_H

#include "config.h"
#include <array>
#include <vector>
#include <memory>
#include <atomic>
#include <mutex>
#include <thread>
#include <chrono>

namespace gobang {

// 硬件资源配置结构
struct HardwareResources {
    // CPU 资源
    int cpu_cores;
    int available_threads;
    size_t total_memory_mb;
    size_t available_memory_mb;
    
    // GPU 资源 (如有)
    bool has_gpu;
    size_t gpu_memory_mb;
    size_t available_gpu_memory_mb;
    float gpu_compute_score;  // GPU 算力评分
    
    // 缓存层级
    size_t l1_cache_size;
    size_t l2_cache_size;
    size_t l3_cache_size;
    
    // NUMA 节点数
    int numa_nodes;
    
    HardwareResources();
};

// 元学习参数空间 - 所有可动态调整的参数
struct MetaParams {
    // MCTS 搜索参数
    int mcts_simulations;           // 模拟次数 [100-100000]
    float mcts_c_puct;              // PUCT 常数 [0.1-5.0]
    float mcts_dirichlet_alpha;     // Dirichlet alpha [0.01-1.0]
    float mcts_noise_scale;         // 噪声缩放 [0.0-1.0]
    int mcts_virtual_loss;          // 虚拟损失 [0-10]
    int mcts_parallel_threads;      // 并行搜索线程数 [1-CPU 核心数]
    
    // 网络推理参数
    bool use_fp16;                  // 是否使用 FP16 推理
    int onnx_intra_threads;         // ONNX 内部线程数 [1-32]
    int onnx_inter_threads;         // ONNX 跨操作线程数 [1-8]
    bool use_cpu_affinity;          // 是否绑定 CPU 亲和性
    
    // 计算并行度
    int batch_size;                 // 批处理大小 [1-64]
    bool enable_tensor_core;        // 启用 Tensor Core(如支持)
    
    // 缓存策略
    int transposition_table_size;   // 置换表大小 (MB) [16-8192]
    bool enable_cache_prefetch;     // 启用缓存预取
    
    // 特征计算
    int feature_planes;             // 特征平面数 [4-20]
    bool enable_dynamic_features;   // 动态特征计算
    
    // 决策温度
    float decision_temperature;     // 决策温度 [0.01-2.0]
    float temperature_decay;        // 温度衰减率 [0.9-0.999]
    
    // 元学习在线微调
    bool enable_online_adaptation;  // 启用在线自适应
    float online_lr;                // 在线学习率 [1e-6 - 1e-2]
    int online_update_freq;         // 在线更新频率 (步数) [10-1000]
    int online_batch_size;          // 在线微调批次 [1-32]
    
    // 内存管理
    bool enable_memory_pool;        // 启用内存池
    size_t memory_pool_size_mb;     // 内存池大小 (MB)
    
    MetaParams();
    
    // 根据硬件资源初始化默认值
    void init_from_hardware(const HardwareResources& hw);
};

// 性能指标
struct PerformanceMetrics {
    // 推理性能
    float inference_time_ms;        // 单次推理耗时 (ms)
    float positions_per_second;     // 每秒处理位置数
    
    // 搜索性能
    float search_depth;             // 平均搜索深度
    float nodes_per_second;         // 每秒搜索节点数
    
    // 内存使用
    size_t memory_used_mb;          // 已用内存 (MB)
    size_t cache_hit_rate;          // 缓存命中率 (%)
    
    // 棋力评估
    float elo_estimate;             // 预估等级分
    float win_rate;                 // 胜率估计
    
    // 硬件利用率
    float cpu_utilization;          // CPU 利用率 (%)
    float gpu_utilization;          // GPU 利用率 (%)
    float memory_bandwidth_util;    // 内存带宽利用率 (%)
    
    PerformanceMetrics();
};

// 元学习状态
enum class MetaLearningState {
    INFERENCE_ONLY,         // 仅推理
    ONLINE_ADAPTATION,      // 在线自适应
    AGGRESSIVE_TUNING,      // 激进调优
    CONSERVATIVE_STABLE     // 保守稳定
};

// 元学习器 - 统一接管所有动态调参
class MetaLearner {
public:
    MetaLearner();
    ~MetaLearner();
    
    // 初始化 (调用一次)
    void initialize(const HardwareResources& hw);
    
    // 获取当前最优参数配置
    const MetaParams& get_current_params() const;
    
    // 根据对局阶段动态调整参数
    void adapt_to_game_stage(
        size_t move_count,
        float current_win_rate,
        float time_remaining_sec = 0.0f
    );
    
    // 根据性能反馈在线微调
    void online_feedback(const PerformanceMetrics& metrics);
    
    // 获取硬件资源
    const HardwareResources& get_hardware() const;
    
    // 设置元学习状态
    void set_state(MetaLearningState state);
    MetaLearningState get_state() const;
    
    // 获取性能指标
    const PerformanceMetrics& get_metrics() const;
    
    // 触发紧急优化 (时间不足时)
    void emergency_optimize(float time_pressure_factor);
    
    // 重置统计
    void reset_stats();
    
private:
    // 硬件资源
    HardwareResources hw_;
    
    // 当前参数
    MetaParams current_params_;
    
    // 性能指标
    PerformanceMetrics metrics_;
    
    // 元学习状态
    MetaLearningState state_;
    
    // 历史性能记录 (用于在线学习)
    std::vector<PerformanceMetrics> history_;
    size_t max_history_size_;
    
    // 同步锁
    mutable std::mutex mutex_;
    
    // 内部方法
    void compute_optimal_params();
    void update_hardware_stats();
    float estimate_elo(const MetaParams& params) const;
    void apply_cpu_affinity();
    void optimize_memory_layout();
    
    // 元学习权重 (简化版，实际应由训练得到)
    std::array<float, 20> meta_weights_;
    bool weights_initialized_;
};

} // namespace gobang

#endif // GOBANG_META_LEARNER_H
