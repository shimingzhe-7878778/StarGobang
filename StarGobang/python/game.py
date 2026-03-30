"""
五子棋游戏逻辑模块

MIT License

Copyright (c) 2026 StarGobang Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

【协议合规】本模块不使用人类棋谱，所有数据均为合成生成，符合 MIT 协议精神
"""
import numpy as np
from typing import Tuple, List, Optional


# 棋盘大小常量
BOARD_SIZE = 15

# 玩家常量
EMPTY = 0  # 空位
BLACK = 1  # 黑棋
WHITE = 2  # 白棋


class Board:
    """15x15 五子棋棋盘"""
    
    def __init__(self):
        """初始化空棋盘"""
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.move_history: List[Tuple[int, int, int]] = []  # (x, y, player)
    
    def reset(self):
        """重置棋盘"""
        self.board.fill(EMPTY)
        self.move_history.clear()
    
    def get_cell(self, x: int, y: int) -> int:
        """获取指定位置的棋子"""
        return self.board[x, y]
    
    def set_cell(self, x: int, y: int, player: int):
        """在指定位置放置棋子"""
        self.board[x, y] = player
    
    def make_move(self, x: int, y: int, player: int):
        """落子并记录历史"""
        if self.board[x, y] == EMPTY:
            self.board[x, y] = player
            self.move_history.append((x, y, player))
    
    def undo_move(self):
        """撤销最后一步"""
        if self.move_history:
            x, y, _ = self.move_history.pop()
            self.board[x, y] = EMPTY
    
    def is_full(self) -> bool:
        """检查棋盘是否已满"""
        return np.all(self.board != EMPTY)
    
    def get_empty_positions(self) -> List[Tuple[int, int]]:
        """获取所有空位"""
        return list(zip(*np.where(self.board == EMPTY)))
    
    def copy(self) -> 'Board':
        """复制棋盘"""
        new_board = Board()
        new_board.board = self.board.copy()
        new_board.move_history = self.move_history.copy()
        return new_board
    
    def current_player(self) -> int:
        """获取当前应落子的玩家（基于历史步数）"""
        if len(self.move_history) % 2 == 0:
            return BLACK
        else:
            return WHITE


def count_direction(board: Board, x: int, y: int, player: int, dx: int, dy: int) -> int:
    """
    统计某方向上的连子数（与 C++ count_direction 逐行对应）
    
    Args:
        board: 棋盘对象
        x, y: 起始坐标
        player: 玩家
        dx, dy: 方向向量
        
    Returns:
        连子数量
    """
    count = 0
    
    # 正方向计数
    nx, ny = x + dx, y + dy
    while (0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and 
           board.get_cell(nx, ny) == player):
        count += 1
        nx += dx
        ny += dy
    
    # 反方向计数
    nx, ny = x - dx, y - dy
    while (0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and 
           board.get_cell(nx, ny) == player):
        count += 1
        nx -= dx
        ny -= dy
    
    return count


def is_live_four(board: Board, x: int, y: int, player: int) -> bool:
    """
    检查是否形成活四（与 C++ is_live_four 逐行对应）
    活四：形成 4 子且两端都为空
    """
    count = 1 + count_direction(board, x, y, player, 1, 0)
    if count == 4:
        # 检查两端是否为空
        left_empty = (x - 1 >= 0 and board.get_cell(x - 1, y) == EMPTY)
        right_empty = (x + 1 < BOARD_SIZE and board.get_cell(x + 1, y) == EMPTY)
        return left_empty and right_empty
    return False


def is_open_four(board: Board, x: int, y: int, player: int) -> bool:
    """
    检查是否形成冲四（与 C++ is_open_four 逐行对应）
    冲四：形成 4 子但至少一端被阻挡
    """
    count = 1 + count_direction(board, x, y, player, 1, 0)
    if count == 4:
        left_empty = (x - 1 >= 0 and board.get_cell(x - 1, y) == EMPTY)
        right_empty = (x + 1 < BOARD_SIZE and board.get_cell(x + 1, y) == EMPTY)
        return not left_empty or not right_empty
    return False


def is_live_three(board: Board, x: int, y: int, player: int) -> bool:
    """
    检查是否形成活三（与 C++ is_live_three 逐行对应）
    活三：形成 3 子且两端都为空
    """
    count = 1 + count_direction(board, x, y, player, 1, 0)
    if count == 3:
        left_empty = (x - 1 >= 0 and board.get_cell(x - 1, y) == EMPTY)
        right_empty = (x + 1 < BOARD_SIZE and board.get_cell(x + 1, y) == EMPTY)
        return left_empty and right_empty
    return False


def is_double_three(board: Board, x: int, y: int, player: int) -> bool:
    """
    检查是否形成双三（与 C++ is_double_three 逐行对应）
    双三：同时在两个或以上方向形成活三
    """
    live_three_count = 0
    
    # 创建临时棋盘用于检测
    temp_board = board.copy()
    temp_board.make_move(x, y, player)
    
    # 垂直方向检查
    vert_count = 1
    ny = y - 1
    while ny >= 0 and temp_board.get_cell(x, ny) == player:
        vert_count += 1
        ny -= 1
    ny = y + 1
    while ny < BOARD_SIZE and temp_board.get_cell(x, ny) == player:
        vert_count += 1
        ny += 1
    
    if vert_count == 3:
        up_empty = (y - 1 >= 0 and temp_board.get_cell(x, y - 1) == EMPTY)
        down_empty = (y + 1 < BOARD_SIZE and temp_board.get_cell(x, y + 1) == EMPTY)
        if up_empty and down_empty:
            live_three_count += 1
    
    # 主对角线检查
    diag1_count = 1
    nx, ny = x - 1, y - 1
    while nx >= 0 and ny >= 0 and temp_board.get_cell(nx, ny) == player:
        diag1_count += 1
        nx -= 1
        ny -= 1
    nx, ny = x + 1, y + 1
    while nx < BOARD_SIZE and ny < BOARD_SIZE and temp_board.get_cell(nx, ny) == player:
        diag1_count += 1
        nx += 1
        ny += 1
    
    if diag1_count == 3:
        up_left_empty = (x - 1 >= 0 and y - 1 >= 0 and 
                        temp_board.get_cell(x - 1, y - 1) == EMPTY)
        down_right_empty = (x + 1 < BOARD_SIZE and y + 1 < BOARD_SIZE and 
                           temp_board.get_cell(x + 1, y + 1) == EMPTY)
        if up_left_empty and down_right_empty:
            live_three_count += 1
    
    # 副对角线检查
    diag2_count = 1
    nx, ny = x + 1, y - 1
    while nx < BOARD_SIZE and ny >= 0 and temp_board.get_cell(nx, ny) == player:
        diag2_count += 1
        nx += 1
        ny -= 1
    nx, ny = x - 1, y + 1
    while nx >= 0 and ny < BOARD_SIZE and temp_board.get_cell(nx, ny) == player:
        diag2_count += 1
        nx -= 1
        ny += 1
    
    if diag2_count == 3:
        up_right_empty = (x + 1 < BOARD_SIZE and y - 1 >= 0 and 
                         temp_board.get_cell(x + 1, y - 1) == EMPTY)
        down_left_empty = (x - 1 >= 0 and y + 1 < BOARD_SIZE and 
                          temp_board.get_cell(x - 1, y + 1) == EMPTY)
        if up_right_empty and down_left_empty:
            live_three_count += 1
    
    return live_three_count >= 2


def is_double_four(board: Board, x: int, y: int, player: int) -> bool:
    """
    检查是否形成双四（与 C++ is_double_four 逐行对应）
    双四：同时在两个或以上方向形成四（活四或冲四）
    """
    four_count = 0
    
    # 创建临时棋盘用于检测
    temp_board = board.copy()
    temp_board.make_move(x, y, player)
    
    # 四个方向检查
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    
    for dx, dy in directions:
        count = 1
        nx, ny = x + dx, y + dy
        while (0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and 
               temp_board.get_cell(nx, ny) == player):
            count += 1
            nx += dx
            ny += dy
        
        nx, ny = x - dx, y - dy
        while (0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and 
               temp_board.get_cell(nx, ny) == player):
            count += 1
            nx -= dx
            ny -= dy
        
        if count >= 4:
            four_count += 1
    
    return four_count >= 2


def is_overline(board: Board, x: int, y: int, player: int) -> bool:
    """
    检查是否形成长连（与 C++ is_overline 逐行对应）
    长连：形成超过 5 子的连线
    """
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    
    for dx, dy in directions:
        count = 1
        nx, ny = x + dx, y + dy
        while (0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and 
               board.get_cell(nx, ny) == player):
            count += 1
            nx += dx
            ny += dy
        
        nx, ny = x - dx, y - dy
        while (0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and 
               board.get_cell(nx, ny) == player):
            count += 1
            nx -= dx
            ny -= dy
        
        if count > 5:
            return True
    
    return False


def is_forbidden_move(board: Board, x: int, y: int, player: int) -> bool:
    """
    检查指定位置是否为禁手（与 C++ is_forbidden_move 逐行对应）
    
    【关键】此函数逻辑必须与C++ 端 forbidden_move_detector.cpp逐行对应
    包含：方向扫描顺序、活三/冲四判定条件
    
    Args:
        board: 棋盘对象
        x, y: 待检查的坐标
        player: 玩家（1=黑棋，2=白棋）
        
    Returns:
        True: 禁手
        False: 非禁手
    """
    # 【重要】白棋没有禁手（与 C++ 第 223-225 行对应）
    if player == WHITE:
        return False
    
    # 黑棋禁手检查
    
    # 1. 长连禁手（与 C++ 第 229-231 行对应）
    if is_overline(board, x, y, player):
        return True
    
    # 2. 双三禁手（与 C++ 第 234-236 行对应）
    if is_double_three(board, x, y, player):
        return True
    
    # 3. 双四禁手（与 C++ 第 239-241 行对应）
    if is_double_four(board, x, y, player):
        return True
    
    return False


def check_win(board: Board, x: int, y: int, player: int) -> bool:
    """
    检查是否获胜（形成五连）
    
    Args:
        board: 棋盘对象
        x, y: 最后落子位置
        player: 玩家
        
    Returns:
        True: 获胜
        False: 未获胜
    """
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    
    for dx, dy in directions:
        count = 1
        
        # 正方向
        nx, ny = x + dx, y + dy
        while (0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and 
               board.get_cell(nx, ny) == player):
            count += 1
            nx += dx
            ny += dy
        
        # 反方向
        nx, ny = x - dx, y - dy
        while (0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and 
               board.get_cell(nx, ny) == player):
            count += 1
            nx -= dx
            ny -= dy
        
        if count >= 5:
            return True
    
    return False


def is_valid_move(board: Board, x: int, y: int, player: int) -> bool:
    """
    检查落子是否合法
    
    Args:
        board: 棋盘对象
        x, y: 待检查的坐标
        player: 玩家
        
    Returns:
        True: 合法
        False: 不合法
    """
    # 检查是否在棋盘内
    if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
        return False
    
    # 检查位置是否为空
    if board.get_cell(x, y) != EMPTY:
        return False
    
    # 检查是否为禁手（仅黑棋）
    if is_forbidden_move(board, x, y, player):
        return False
    
    return True


def get_legal_moves(board: Board, player: int) -> List[Tuple[int, int]]:
    """
    获取所有合法落子位置
    
    Args:
        board: 棋盘对象
        player: 玩家
        
    Returns:
        合法位置列表
    """
    legal_moves = []
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            if is_valid_move(board, x, y, player):
                legal_moves.append((x, y))
    return legal_moves
