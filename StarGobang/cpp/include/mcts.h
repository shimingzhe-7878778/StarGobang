// [合规声明] 1. 纯推理组件无交互逻辑 2. NN 推理 100% 通过 ONNX Runtime (MIT)
// [架构升级] 元学习动态调整 MCTS 参数 + 并行搜索
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
#ifndef GOBANG_MCTS_H
#define GOBANG_MCTS_H

#include "board.h"
#include "config.h"
#include "meta_learner.h"
#include <vector>
#include <memory>
#include <random>
#include <atomic>
#include <mutex>

namespace gobang {

// 前向声明
class GobangEngine;

// MCTS 节点（支持并行访问）
class MCTSNode {
public:
    explicit MCTSNode(const Board& board);
    
    // 选择最佳子节点（线程安全）
    MCTSNode* select_child(float c_puct);
    
    // 扩展节点
    void expand(const std::vector<float>& policy_probs, const MCTSConfig& config);
    
    // 反向传播（线程安全，支持虚拟损失）
    void backup(float value, int virtual_loss);
    
    // 获取访问次数
    int get_visit_count() const { return visit_count_.load(); }
    
    // 获取 Q 值
    float get_q_value() const;
    
    // 检查是否展开
    bool is_expanded() const { return expanded_.load(); }
    
    // 获取子节点
    std::vector<std::unique_ptr<MCTSNode>>& get_children() { return children_; }
    
    // 获取动作概率先验
    float get_prior(int move_idx) const;
    
    // 获取移动索引（公开方法）
    int get_move_index(size_t idx) const {
        if (idx < move_indices_.size()) {
            return move_indices_[idx];
        }
        return -1;
    }
    
    // 原子操作支持
    std::atomic<int>& get_atomic_visit_count() { return visit_count_; }
    std::atomic<float>& get_atomic_value_sum() { return value_sum_; }
    
private:
    Board board_;
    std::vector<std::unique_ptr<MCTSNode>> children_;
    std::vector<float> priors_;
    std::vector<int> move_indices_;
    std::atomic<int> visit_count_;           // 原子计数器
    std::atomic<float> value_sum_;           // 原子累加器
    std::atomic<bool> expanded_;             // 原子标志
    mutable std::mutex node_mutex_;          // 节点锁
};

// MCTS 搜索引擎（支持并行搜索与动态参数调整）
class MCTS {
public:
    // 构造函数：由元学习器自动配置参数
    explicit MCTS(GobangEngine* engine, MetaLearner* meta_learner);
    
    // 执行搜索并返回最佳移动
    std::pair<int, int> search(const Board& root_board);
    
    // 获取根节点
    MCTSNode* get_root() { return root_.get(); }
    
    // ========== 并行搜索接口 ==========
    
    // 并行搜索（多线程版本）
    std::pair<int, int> parallel_search(const Board& root_board, int num_threads);
    
private:
    // 模拟一次游戏
    float simulate(MCTSNode* node);
    
    // 评估局面（调用神经网络）
    float evaluate(const Board& board);
    
    // 添加 Dirichlet 噪声
    void add_dirichlet_noise(std::vector<float>& probs, const MCTSConfig& config);
    
    // 单次模拟（用于并行）
    void single_simulation();
    
    GobangEngine* engine_;
    MetaLearner* meta_learner_;  // 元学习器引用
    MCTSConfig config_;          // 当前配置（由元学习动态调整）
    std::unique_ptr<MCTSNode> root_;
    std::mt19937 rng_;
    
    // 并行搜索同步
    std::atomic<int> simulation_count_;  // 已完成模拟次数
    std::atomic<bool> search_done_;      // 搜索完成标志
};

} // namespace gobang

#endif // GOBANG_MCTS_H
