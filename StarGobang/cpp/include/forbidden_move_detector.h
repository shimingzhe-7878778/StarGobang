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
#ifndef GOBANG_FORBIDDEN_MOVE_DETECTOR_H
#define GOBANG_FORBIDDEN_MOVE_DETECTOR_H

#include "board.h"

namespace gobang {

// 禁手检测器（仅用于黑棋，白棋无禁手）
class ForbiddenMoveDetector {
public:
    // 检查指定位置是否为禁手
    // 返回值：true 表示禁手，false 表示非禁手
    static bool is_forbidden_move(const Board& board, int x, int y, Player player);
    
private:
    // 统计某方向上的连子数
    static int count_direction(
        const Board& board, 
        int x, int y, 
        Player player,
        int dx, int dy
    );
    
    // 检查是否形成活四
    static bool is_live_four(const Board& board, int x, int y, Player player);
    
    // 检查是否形成冲四
    static bool is_open_four(const Board& board, int x, int y, Player player);
    
    // 检查是否形成活三
    static bool is_live_three(const Board& board, int x, int y, Player player);
    
    // 检查是否形成双三
    static bool is_double_three(const Board& board, int x, int y, Player player);
    
    // 检查是否形成双四
    static bool is_double_four(const Board& board, int x, int y, Player player);
    
    // 检查是否形成长连（超过 5 子）
    static bool is_overline(const Board& board, int x, int y, Player player);
};

} // namespace gobang

#endif // GOBANG_FORBIDDEN_MOVE_DETECTOR_H
