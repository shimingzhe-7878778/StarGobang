// [合规声明] 1. 纯推理组件无交互逻辑 2. NN 推理 100% 通过 ONNX Runtime (MIT)
// [架构升级] 集成在线自适应与元学习监控
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
#include "learning_monitor.h"
#include <fstream>
#include <cstring>
#include <algorithm>
#include <numeric>

namespace gobang {

LearningMonitor::LearningMonitor() 
    : stats_{}, max_perf_history_size_(100) {
    reset();
}

LearningMonitor::~LearningMonitor() = default;

void LearningMonitor::record_game(
    const std::vector<Board>& positions,
    const std::vector<std::pair<int, int>>& moves,
    GameResult result
) {
    if (positions.empty() || positions.size() != moves.size()) {
        return;
    }
    
    // 为每个局面生成训练样本
    for (size_t i = 0; i < positions.size(); ++i) {
        TrainingSample sample;
        
        // 填充特征
        auto features = positions[i].create_feature_tensor();
        sample.features = features;
        
        // 生成 policy（one-hot 编码）
        std::memset(sample.policy.data(), 0, sizeof(sample.policy));
        int move_idx = moves[i].second * BOARD_SIZE + moves[i].first;
        if (move_idx >= 0 && move_idx < BOARD_CELLS) {
            sample.policy[move_idx] = 1.0f;
        }
        
        // 设置 value
        sample.value = static_cast<float>(static_cast<int>(result));
        
        // 计算预测误差（如果有元学习器）
        if (meta_learner_) {
            compute_prediction_error(sample);
        }
        
        samples_.push_back(sample);
    }
    
    // 更新统计
    stats_.total_games++;
    stats_.total_positions += positions.size();
    
    if (result == GameResult::BLACK_WIN) {
        stats_.black_win_rate = (stats_.black_win_rate * (stats_.total_games - 1) + 1.0f) / 
                                static_cast<float>(stats_.total_games);
    } else if (result == GameResult::WHITE_WIN) {
        stats_.white_win_rate = (stats_.white_win_rate * (stats_.total_games - 1) + 1.0f) / 
                                static_cast<float>(stats_.total_games);
    } else if (result == GameResult::DRAW) {
        stats_.draw_rate = (stats_.draw_rate * (stats_.total_games - 1) + 1.0f) / 
                          static_cast<float>(stats_.total_games);
    }
    
    // 更新运行统计
    update_running_stats();
}

std::vector<TrainingSample> LearningMonitor::generate_training_data() const {
    return samples_;
}

LearningMonitor::Stats LearningMonitor::get_stats() const {
    return stats_;
}

void LearningMonitor::reset() {
    samples_.clear();
    stats_ = Stats{0, 0, 0.0f, 0.0f, 0.0f};
}

bool LearningMonitor::export_to_file(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    // 写入样本数量
    uint64_t count = samples_.size();
    file.write(reinterpret_cast<const char*>(&count), sizeof(count));
    
    // 写入所有样本
    for (const auto& sample : samples_) {
        file.write(reinterpret_cast<const char*>(sample.features.data()), 
                   sizeof(sample.features));
        file.write(reinterpret_cast<const char*>(sample.policy.data()), 
                   sizeof(sample.policy));
        file.write(reinterpret_cast<const char*>(&sample.value), sizeof(sample.value));
    }
    
    return true;
}

void LearningMonitor::set_meta_learner(MetaLearner* meta_learner) {
    meta_learner_ = meta_learner;
}

void LearningMonitor::online_adapt_step(
    const std::vector<TrainingSample>& batch,
    float learning_rate
) {
    if (!meta_learner_ || batch.empty()) {
        return;
    }
    
    // 简化的在线梯度下降
    // 实际应该更新神经网络权重，这里仅做演示
    
    float total_error = 0.0f;
    for (const auto& sample : batch) {
        total_error += std::abs(sample.prediction_error);
    }
    
    float avg_error = total_error / batch.size();
    
    // 更新统计
    stats_.avg_prediction_error = avg_error;
    stats_.online_updates++;
    
    // 根据误差调整学习率
    if (avg_error > 0.5f) {
        // 误差大，增加学习率
        learning_rate *= 1.1f;
    } else if (avg_error < 0.1f) {
        // 误差小，减小学习率以稳定
        learning_rate *= 0.9f;
    }
    
    // 记录性能反馈
    PerformanceMetrics metrics;
    metrics.inference_time_ms = avg_error * 10.0f;  // 简化映射
    meta_learner_->online_feedback(metrics);
}

PerformanceMetrics LearningMonitor::evaluate_performance() {
    PerformanceMetrics metrics;
    
    // 计算平均预测误差
    if (!samples_.empty()) {
        float total_error = 0.0f;
        for (const auto& sample : samples_) {
            total_error += std::abs(sample.prediction_error);
        }
        metrics.inference_time_ms = total_error / samples_.size() * 10.0f;
    }
    
    // 估计 Elo
    metrics.elo_estimate = 1500.0f + (1.0f - stats_.avg_prediction_error) * 500.0f;
    
    // 胜率
    metrics.win_rate = stats_.black_win_rate;
    
    return metrics;
}

void LearningMonitor::record_performance_feedback(const PerformanceMetrics& metrics) {
    perf_history_.push_back(metrics);
    
    // 限制历史记录大小
    if (perf_history_.size() > max_perf_history_size_) {
        perf_history_.erase(perf_history_.begin());
    }
}

float LearningMonitor::get_performance_trend() const {
    if (perf_history_.size() < 10) {
        return 0.0f;  // 数据不足
    }
    
    // 计算最近 5 次和前 5 次的平均性能差异
    float recent_avg = 0.0f;
    float old_avg = 0.0f;
    
    for (size_t i = 0; i < 5; ++i) {
        recent_avg += perf_history_[perf_history_.size() - 1 - i].elo_estimate;
        old_avg += perf_history_[i].elo_estimate;
    }
    
    recent_avg /= 5.0f;
    old_avg /= 5.0f;
    
    return recent_avg - old_avg;  // >0 表示提升
}

void LearningMonitor::compute_prediction_error(TrainingSample& sample) {
    // 简化的预测误差计算
    // 实际应该使用神经网络预测值与真实值的差异
    
    // 这里假设有一个简单的预测模型
    float predicted_value = 0.5f;  // 简化为随机猜测
    sample.prediction_error = std::abs(sample.value - predicted_value);
    
    // 计算重要性权重
    sample.importance_weight = 1.0f / (1.0f + sample.prediction_error);
}

void LearningMonitor::update_running_stats() {
    // 更新运行统计信息
    if (!samples_.empty()) {
        float total_error = 0.0f;
        for (const auto& sample : samples_) {
            total_error += sample.prediction_error;
        }
        stats_.avg_prediction_error = total_error / samples_.size();
    }
}

} // namespace gobang
