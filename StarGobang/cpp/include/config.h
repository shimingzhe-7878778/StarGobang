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
#ifndef GOBANG_CONFIG_H
#define GOBANG_CONFIG_H

#include <cstddef>
#include <cstdint>

namespace gobang {

// 棋盘尺寸
constexpr int BOARD_SIZE = 15;

// 棋盘状态总数（15*15=225）
constexpr int BOARD_CELLS = BOARD_SIZE * BOARD_SIZE;

// 玩家类型
enum class Player : int {
    NONE = 0,
    BLACK = 1,
    WHITE = 2
};

// 游戏结果
enum class GameResult {
    BLACK_WIN = 1,
    WHITE_WIN = -1,
    DRAW = 0,
    UNKNOWN = 0
};

// MCTS 配置参数 (基础默认值，实际由元学习动态调整)
struct MCTSConfig {
    int num_simulations = 800;          // 模拟次数 (动态范围：100-100000)
    float c_puct = 1.5f;                // PUCT 探索常数 (动态范围：0.1-5.0)
    float dirichlet_alpha = 0.3f;       // Dirichlet 噪声 alpha (动态范围：0.01-1.0)
    float noise_scale = 0.25f;          // 噪声缩放因子 (动态范围：0.0-1.0)
    int virtual_loss = 3;               // 虚拟损失值 (动态范围：0-10)
    int parallel_threads = 1;           // 并行搜索线程数 (由元学习根据 CPU 核心数设置)
    
    // 动态参数标志
    bool dynamic_adjustment = true;     // 启用动态调整
    int min_simulations = 100;          // 最小模拟次数
    int max_simulations = 100000;       // 最大模拟次数
};

// 神经网络输入输出规格
struct NetworkSpec {
    // 输入张量形状：[batch, channels, height, width]
    static constexpr size_t BATCH_SIZE = 1;
    static constexpr size_t INPUT_CHANNELS = 10;  // 可由元学习动态调整 [4-20]
    static constexpr size_t HEIGHT = BOARD_SIZE;
    static constexpr size_t WIDTH = BOARD_SIZE;
    
    // 输出维度
    static constexpr size_t POLICY_OUTPUT = BOARD_CELLS;  // 225 个落子点概率
    static constexpr size_t VALUE_OUTPUT = 1;              // 单个胜率估计
};

// 元学习配置
struct MetaLearningConfig {
    bool enabled = true;                    // 启用元学习
    bool online_adaptation = true;          // 在线自适应
    float online_learning_rate = 1e-4f;     // 在线学习率
    int adaptation_frequency = 10;          // 自适应频率 (每 N 步)
    int history_size = 100;                 // 历史记录大小
    
    // 硬件感知配置
    bool hardware_aware = true;             // 硬件感知
    bool auto_thread_binding = true;        // 自动线程绑定
    bool memory_pool_enabled = true;        // 内存池启用
    
    // 性能模式
    enum class PerformanceMode {
        MAX_STRENGTH,                       // 极限棋力 (优先)
        BALANCED,                           // 平衡
        FAST                                // 快速
    };
    PerformanceMode mode = PerformanceMode::MAX_STRENGTH;
};

} // namespace gobang

#endif // GOBANG_CONFIG_H
