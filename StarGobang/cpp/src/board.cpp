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
#include "board.h"
#include <cstring>

namespace gobang {

Board::Board() : current_player_(Player::BLACK) {
    // 初始化为空棋盘
    for (auto& row : grid_) {
        for (auto& cell : row) {
            cell = Player::NONE;
        }
    }
}

Player Board::get_cell(int x, int y) const {
    if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE) {
        return Player::NONE;
    }
    return grid_[y][x];
}

void Board::make_move(int x, int y, Player player) {
    if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE) {
        return;
    }
    if (grid_[y][x] != Player::NONE) {
        return;
    }
    
    grid_[y][x] = player;
    history_.push_back({x, y, player});
    current_player_ = (player == Player::BLACK) ? Player::WHITE : Player::BLACK;
}

void Board::undo_move() {
    if (history_.empty()) {
        return;
    }
    
    const Move& last = history_.back();
    grid_[last.y][last.x] = Player::NONE;
    current_player_ = last.player;
    history_.pop_back();
}

bool Board::is_empty(int x, int y) const {
    if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE) {
        return false;
    }
    return grid_[y][x] == Player::NONE;
}

bool Board::is_full() const {
    return history_.size() >= static_cast<size_t>(BOARD_CELLS);
}

Player Board::current_player() const {
    return current_player_;
}

const Move* Board::last_move() const {
    if (history_.empty()) {
        return nullptr;
    }
    return &history_.back();
}

const std::vector<Move>& Board::get_history() const {
    return history_;
}

size_t Board::move_count() const {
    return history_.size();
}

std::array<float, NetworkSpec::INPUT_CHANNELS * BOARD_SIZE * BOARD_SIZE> 
Board::create_feature_tensor() const {
    std::array<float, NetworkSpec::INPUT_CHANNELS * BOARD_SIZE * BOARD_SIZE> features{};
    
    // 特征平面说明（10 通道）：
    // [0-3]: 黑棋历史位置（最近 4 步）
    // [4-7]: 白棋历史位置（最近 4 步）
    // [8]: 当前玩家颜色（全 1 为黑，全 0 为白）
    // [9]: 对手颜色（全 1 为白，全 0 为黑）
    
    const auto& history = get_history();
    size_t total_cells = BOARD_SIZE * BOARD_SIZE;
    
    // 初始化所有特征为 0
    std::memset(features.data(), 0, sizeof(features));
    
    // 填充最近 4 步的黑棋位置
    for (int i = 0; i < 4 && i < static_cast<int>(history.size()); ++i) {
        const Move& move = history[history.size() - 1 - i];
        if (move.player == Player::BLACK) {
            size_t idx = i * total_cells + move.y * BOARD_SIZE + move.x;
            features[idx] = 1.0f;
        }
    }
    
    // 填充最近 4 步的白棋位置
    for (int i = 0; i < 4 && i < static_cast<int>(history.size()); ++i) {
        const Move& move = history[history.size() - 1 - i];
        if (move.player == Player::WHITE) {
            size_t idx = (4 + i) * total_cells + move.y * BOARD_SIZE + move.x;
            features[idx] = 1.0f;
        }
    }
    
    // 填充当前玩家颜色
    if (current_player_ == Player::BLACK) {
        for (size_t i = 0; i < total_cells; ++i) {
            features[8 * total_cells + i] = 1.0f;
        }
    } else {
        for (size_t i = 0; i < total_cells; ++i) {
            features[9 * total_cells + i] = 1.0f;
        }
    }
    
    return features;
}

} // namespace gobang
