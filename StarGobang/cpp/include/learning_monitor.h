// [合规声明] 1. 纯推理组件无交互逻辑 2. NN 推理 100% 通过 ONNX Runtime (MIT)
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
#ifndef GOBANG_LEARNING_MONITOR_H
#define GOBANG_LEARNING_MONITOR_H

#include "board.h"
#include "config.h"
#include "meta_learner.h"
#include <vector>
#include <string>
#include <cstdint>
#include <functional>

namespace gobang {

// 训练样本数据结构
struct TrainingSample {
    std::array<float, NetworkSpec::INPUT_CHANNELS * BOARD_SIZE * BOARD_SIZE> features;
    std::array<float, BOARD_CELLS> policy;  // 落子概率分布
    float value;                             // 局面价值 [-1, 1]
    
    // 在线自适应用
    float prediction_error = 0.0f;           // 预测误差
    float importance_weight = 1.0f;          // 重要性权重
};

// 学习监控器（用于数据记录和统计 + 在线自适应）
class LearningMonitor {
public:
    LearningMonitor();
    ~LearningMonitor();
    
    // 记录对局数据
    void record_game(
        const std::vector<Board>& positions,
        const std::vector<std::pair<int, int>>& moves,
        GameResult result
    );
    
    // 生成训练数据
    std::vector<TrainingSample> generate_training_data() const;
    
    // 统计数据
    struct Stats {
        uint64_t total_games;
        uint64_t total_positions;
        float black_win_rate;
        float white_win_rate;
        float draw_rate;
        
        // 在线自适应统计
        float avg_prediction_error;
        float adaptation_improvement;  // 自适应改进幅度
        uint64_t online_updates;        // 在线更新次数
    };
    
    Stats get_stats() const;
    
    // 重置统计
    void reset();
    
    // 导出到文件（可选功能）
    bool export_to_file(const std::string& filename) const;
    
    // ========== 在线自适应功能 ==========
    
    // 设置元学习器引用
    void set_meta_learner(MetaLearner* meta_learner);
    
    // 在线微调 (由元学习调用)
    void online_adapt_step(
        const std::vector<TrainingSample>& batch,
        float learning_rate
    );
    
    // 评估模型性能
    PerformanceMetrics evaluate_performance();
    
    // 记录性能反馈
    void record_performance_feedback(const PerformanceMetrics& metrics);
    
    // 获取最近性能趋势
    float get_performance_trend() const;  // >0 提升，<0 下降
    
private:
    std::vector<TrainingSample> samples_;
    Stats stats_;
    
    // 元学习器引用
    MetaLearner* meta_learner_ = nullptr;
    
    // 性能历史记录
    std::vector<PerformanceMetrics> perf_history_;
    size_t max_perf_history_size_ = 100;
    
    // 内部方法
    void compute_prediction_error(TrainingSample& sample);
    void update_running_stats();
};

} // namespace gobang

#endif // GOBANG_LEARNING_MONITOR_H
