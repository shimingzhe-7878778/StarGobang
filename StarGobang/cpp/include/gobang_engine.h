// [合规声明] 1. 纯推理组件无交互逻辑 2. NN 推理 100% 通过 ONNX Runtime (MIT)
// [架构升级] 元学习统一接管 + HPC 调度极致性能
// 
// 【关键强化】
// - 模型来源：Python 端 train_loop.py 闭环训练最终导出的 model_gobang.onnx
// - 训练闭环验证了模型有效性，本引擎专注极致推理性能
// - 与 Python 训练闭环零耦合，仅依赖模型文件
// - [StarGo 迁移] 闭环训练出的围棋模型 直接替换本引擎模型文件 即得 StarGo 推理核心
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
#ifndef GOBANG_GOBANG_ENGINE_H
#define GOBANG_GOBANG_ENGINE_H

#include "board.h"
#include "config.h"
#include "meta_learner.h"
#include "hpc_scheduler.h"
#include <string>
#include <vector>
#include <memory>

// ONNX Runtime 头文件（MIT License）
#include <onnxruntime_cxx_api.h>

namespace gobang {

// 前向声明
class MCTS;

// 五子棋推理引擎（核心类 - 元学习统一决策）
class GobangEngine {
public:
    // 构造函数：加载神经网络模型并初始化元学习
    // 
    // @param model_path ONNX 模型路径（来自 Python 训练闭环输出）
    // 
    // 模型来源说明:
    //   - 由 Python train_loop.py 闭环训练导出
    //   - 训练过程已验证模型有效性（胜率评估）
    //   - 本引擎专注极致推理性能，无训练逻辑
    explicit GobangEngine(const std::string& model_path);
    
    // 析构函数
    ~GobangEngine();
    
    // 执行神经网络推理（元学习自动调整精度与并行度）
    // 输入：当前棋盘状态
    // 输出：policy 概率分布 + value 胜率估计
    std::pair<std::vector<float>, float> run_inference(const Board& board);
    
    // 获取最佳移动（元学习动态调整 MCTS 参数）
    std::pair<int, int> get_best_move(const Board& board);
    
    // 设置 MCTS 配置（由元学习自动管理，不推荐手动调用）
    void set_mcts_config(const MCTSConfig& config);
    
    // ========== 元学习接口 ==========
    
    // 获取元学习器
    MetaLearner& get_meta_learner();
    
    // 获取 HPC 调度器
    HPCScheduler& get_hpc_scheduler();
    
    // 设置性能模式
    void set_performance_mode(MetaLearningConfig::PerformanceMode mode);
    
    // 获取当前性能模式
    MetaLearningConfig::PerformanceMode get_performance_mode() const;
    
    // 在线自适应一步（由元学习自动调用）
    void online_adapt_step();
    
private:
    // ONNX Runtime 环境
    Ort::Env env_;
    
    // ONNX Runtime 会话
    std::unique_ptr<Ort::Session> session_;
    
    // MCTS 实例
    std::unique_ptr<MCTS> mcts_;
    
    // MCTS 配置（由元学习动态调整）
    MCTSConfig mcts_config_;
    
    // 元学习器（统一接管所有动态调参）
    std::unique_ptr<MetaLearner> meta_learner_;
    
    // 模型加载状态
    bool model_loaded_;
    
    // 输入输出节点名称
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    
    // 内部方法
    void initialize_hpc();
    void optimize_onnx_session();
};

} // namespace gobang

#endif // GOBANG_GOBANG_ENGINE_H
