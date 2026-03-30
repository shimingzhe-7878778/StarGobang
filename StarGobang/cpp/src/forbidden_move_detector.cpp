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
#include "forbidden_move_detector.h"

namespace gobang {

int ForbiddenMoveDetector::count_direction(
    const Board& board, 
    int x, int y, 
    Player player,
    int dx, int dy
) {
    int count = 0;
    
    // 正方向计数
    int nx = x + dx;
    int ny = y + dy;
    while (nx >= 0 && nx < BOARD_SIZE && ny >= 0 && ny < BOARD_SIZE &&
           board.get_cell(nx, ny) == player) {
        ++count;
        nx += dx;
        ny += dy;
    }
    
    // 反方向计数
    nx = x - dx;
    ny = y - dy;
    while (nx >= 0 && nx < BOARD_SIZE && ny >= 0 && ny < BOARD_SIZE &&
           board.get_cell(nx, ny) == player) {
        ++count;
        nx -= dx;
        ny -= dy;
    }
    
    return count;
}

bool ForbiddenMoveDetector::is_live_four(const Board& board, int x, int y, Player player) {
    // 活四：形成 4 子且两端都为空
    int count = 1 + count_direction(board, x, y, player, 1, 0);
    if (count == 4) {
        // 检查两端是否为空
        bool left_empty = (x - 1 >= 0 && board.get_cell(x - 1, y) == Player::NONE);
        bool right_empty = (x + 1 < BOARD_SIZE && board.get_cell(x + 1, y) == Player::NONE);
        return left_empty && right_empty;
    }
    return false;
}

bool ForbiddenMoveDetector::is_open_four(const Board& board, int x, int y, Player player) {
    // 冲四：形成 4 子但至少一端被阻挡
    int count = 1 + count_direction(board, x, y, player, 1, 0);
    if (count == 4) {
        bool left_empty = (x - 1 >= 0 && board.get_cell(x - 1, y) == Player::NONE);
        bool right_empty = (x + 1 < BOARD_SIZE && board.get_cell(x + 1, y) == Player::NONE);
        return !left_empty || !right_empty;
    }
    return false;
}

bool ForbiddenMoveDetector::is_live_three(const Board& board, int x, int y, Player player) {
    // 活三：形成 3 子且两端都为空
    int count = 1 + count_direction(board, x, y, player, 1, 0);
    if (count == 3) {
        bool left_empty = (x - 1 >= 0 && board.get_cell(x - 1, y) == Player::NONE);
        bool right_empty = (x + 1 < BOARD_SIZE && board.get_cell(x + 1, y) == Player::NONE);
        return left_empty && right_empty;
    }
    return false;
}

bool ForbiddenMoveDetector::is_double_three(const Board& board, int x, int y, Player player) {
    // 双三：同时在两个或以上方向形成活三
    int live_three_count = 0;
    
    Board temp_board = board;
    temp_board.make_move(x, y, player);
    
    // 垂直
    int vert_count = 1;
    int ny = y - 1;
    while (ny >= 0 && temp_board.get_cell(x, ny) == player) {
        vert_count++;
        ny--;
    }
    ny = y + 1;
    while (ny < BOARD_SIZE && temp_board.get_cell(x, ny) == player) {
        vert_count++;
        ny++;
    }
    if (vert_count == 3) {
        bool up_empty = (y - 1 >= 0 && temp_board.get_cell(x, y - 1) == Player::NONE);
        bool down_empty = (y + 1 < BOARD_SIZE && temp_board.get_cell(x, y + 1) == Player::NONE);
        if (up_empty && down_empty) {
            live_three_count++;
        }
    }
    
    // 主对角线
    int diag1_count = 1;
    int nx = x - 1;
    ny = y - 1;
    while (nx >= 0 && ny >= 0 && temp_board.get_cell(nx, ny) == player) {
        diag1_count++;
        nx--;
        ny--;
    }
    nx = x + 1;
    ny = y + 1;
    while (nx < BOARD_SIZE && ny < BOARD_SIZE && temp_board.get_cell(nx, ny) == player) {
        diag1_count++;
        nx++;
        ny++;
    }
    if (diag1_count == 3) {
        bool up_left_empty = (x - 1 >= 0 && y - 1 >= 0 && 
                             temp_board.get_cell(x - 1, y - 1) == Player::NONE);
        bool down_right_empty = (x + 1 < BOARD_SIZE && y + 1 < BOARD_SIZE && 
                                temp_board.get_cell(x + 1, y + 1) == Player::NONE);
        if (up_left_empty && down_right_empty) {
            live_three_count++;
        }
    }
    
    // 副对角线
    int diag2_count = 1;
    nx = x + 1;
    ny = y - 1;
    while (nx < BOARD_SIZE && ny >= 0 && temp_board.get_cell(nx, ny) == player) {
        diag2_count++;
        nx++;
        ny--;
    }
    nx = x - 1;
    ny = y + 1;
    while (nx >= 0 && ny < BOARD_SIZE && temp_board.get_cell(nx, ny) == player) {
        diag2_count++;
        nx--;
        ny++;
    }
    if (diag2_count == 3) {
        bool up_right_empty = (x + 1 < BOARD_SIZE && y - 1 >= 0 && 
                              temp_board.get_cell(x + 1, y - 1) == Player::NONE);
        bool down_left_empty = (x - 1 >= 0 && y + 1 < BOARD_SIZE && 
                               temp_board.get_cell(x - 1, y + 1) == Player::NONE);
        if (up_right_empty && down_left_empty) {
            live_three_count++;
        }
    }
    
    return live_three_count >= 2;
}

bool ForbiddenMoveDetector::is_double_four(const Board& board, int x, int y, Player player) {
    // 双四：同时在两个或以上方向形成四（活四或冲四）
    int four_count = 0;
    
    Board temp_board = board;
    temp_board.make_move(x, y, player);
    
    // 四个方向检查
    int directions[4][2] = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};
    
    for (const auto& dir : directions) {
        int count = 1;
        int nx = x + dir[0];
        int ny = y + dir[1];
        while (nx >= 0 && nx < BOARD_SIZE && ny >= 0 && ny < BOARD_SIZE &&
               temp_board.get_cell(nx, ny) == player) {
            count++;
            nx += dir[0];
            ny += dir[1];
        }
        nx = x - dir[0];
        ny = y - dir[1];
        while (nx >= 0 && nx < BOARD_SIZE && ny >= 0 && ny < BOARD_SIZE &&
               temp_board.get_cell(nx, ny) == player) {
            count++;
            nx -= dir[0];
            ny -= dir[1];
        }
        
        if (count >= 4) {
            four_count++;
        }
    }
    
    return four_count >= 2;
}

bool ForbiddenMoveDetector::is_overline(const Board& board, int x, int y, Player player) {
    // 长连：形成超过 5 子的连线
    int directions[4][2] = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};
    
    for (const auto& dir : directions) {
        int count = 1;
        int nx = x + dir[0];
        int ny = y + dir[1];
        while (nx >= 0 && nx < BOARD_SIZE && ny >= 0 && ny < BOARD_SIZE &&
               board.get_cell(nx, ny) == player) {
            count++;
            nx += dir[0];
            ny += dir[1];
        }
        nx = x - dir[0];
        ny = y - dir[1];
        while (nx >= 0 && nx < BOARD_SIZE && ny >= 0 && ny < BOARD_SIZE &&
               board.get_cell(nx, ny) == player) {
            count++;
            nx -= dir[0];
            ny -= dir[1];
        }
        
        if (count > 5) {
            return true;
        }
    }
    
    return false;
}

bool ForbiddenMoveDetector::is_forbidden_move(const Board& board, int x, int y, Player player) {
    // 白棋没有禁手
    if (player == Player::WHITE) {
        return false;
    }
    
    // 黑棋禁手检查
    // 1. 长连禁手
    if (is_overline(board, x, y, player)) {
        return true;
    }
    
    // 2. 双三禁手
    if (is_double_three(board, x, y, player)) {
        return true;
    }
    
    // 3. 双四禁手
    if (is_double_four(board, x, y, player)) {
        return true;
    }
    
    return false;
}

} // namespace gobang
