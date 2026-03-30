// [合规声明] 1. 纯推理组件无交互逻辑 2. NN 推理 100% 通过ONNX Runtime (MIT)
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
#ifndef GOBANG_BOARD_H
#define GOBANG_BOARD_H

#include "config.h"
#include <array>
#include <vector>
#include <cstdint>

namespace gobang {

// 落子记录
struct Move {
    int x;
    int y;
    Player player;
};

// 棋盘类（无状态数据容器）
class Board {
public:
    Board();
    
    // 获取指定位置的玩家
    Player get_cell(int x, int y) const;
    
    // 在指定位置落子
    void make_move(int x, int y, Player player);
    
    // 撤销落子
    void undo_move();
    
    // 检查位置是否为空
    bool is_empty(int x, int y) const;
    
    // 检查棋盘是否已满
    bool is_full() const;
    
    // 获取当前应落子的玩家
    Player current_player() const;
    
    // 获取最后一步落子
    const Move* last_move() const;
    
    // 获取所有历史落子
    const std::vector<Move>& get_history() const;
    
    // 获取步数
    size_t move_count() const;
    
    // 生成特征平面（用于神经网络输入）
    std::array<float, NetworkSpec::INPUT_CHANNELS * BOARD_SIZE * BOARD_SIZE> 
    create_feature_tensor() const;
    
private:
    std::array<std::array<Player, BOARD_SIZE>, BOARD_SIZE> grid_;
    std::vector<Move> history_;
    Player current_player_;
};

} // namespace gobang

#endif // GOBANG_BOARD_H
