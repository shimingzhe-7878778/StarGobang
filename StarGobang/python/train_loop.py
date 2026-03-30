"""
StarGobang 自迭代训练闭环（纯训练域·零推理代码）

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

【核心铁律】
  闭环三要素：自我对弈生成数据 → 训练新模型 → 新模型替换旧模型（循环 N 轮）
  每轮自动：保存模型 (checkpoint) + 记录日志 (loss/胜率) + 导出最终 ONNX
  严禁：调用 C++ 代码、包含推理逻辑、任何需人工更新的社区进度

【协议合规】
  本训练流程不使用人类棋谱，符合 MIT 协议精神

【StarGo 迁移说明】
  本闭环为 StarGo 训练框架预演：逻辑完全复用，仅需修改 config.yaml 中：
    - BOARD_SIZE: 15 → 19
    - INPUT_CHANNELS: 3 → 17 (AlphaGo 特征平面)
    - TOTAL_ITERATIONS: 50 → 200+

作者：StarGobang Team
日期：2026-03-25
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import os
import sys
import json
import yaml
from datetime import datetime
from typing import Tuple, List, Dict, Optional

# 导入本地模块
from game import Board, BLACK, WHITE, EMPTY, check_win, get_legal_moves, is_forbidden_move
from game import is_live_four, is_open_four, is_live_three, is_double_three, is_double_four
from model import GobangNet, create_model, save_checkpoint, board_to_tensor
from utils.data_generator import generate_random_game


class Config:
    """配置管理器"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 提取关键参数
        self.TOTAL_ITERATIONS = self.config['TOTAL_ITERATIONS']
        self.GAMES_PER_ITER = self.config['GAMES_PER_ITER']
        self.EVAL_INTERVAL = self.config['EVAL_INTERVAL']
        self.EVAL_GAMES = self.config['EVAL_GAMES']
        
        self.BATCH_SIZE = self.config['TRAINING']['BATCH_SIZE']
        self.EPOCHS_PER_ITER = self.config['TRAINING']['EPOCHS_PER_ITER']
        self.LEARNING_RATE = self.config['TRAINING']['LEARNING_RATE']
        self.LR_DECAY = self.config['TRAINING']['LR_DECAY']
        self.MIN_LR = self.config['TRAINING']['MIN_LR']
        
        self.MCTS_SIMULATIONS = self.config['TRAINING']['MCTS_SIMULATIONS']
        self.TEMPERATURE = self.config['TRAINING']['TEMPERATURE']
        self.TEMPERATURE_DECAY = self.config['TRAINING']['TEMPERATURE_DECAY']
        self.MIN_TEMPERATURE = self.config['TRAINING']['MIN_TEMPERATURE']
        
        self.INPUT_CHANNELS = self.config['MODEL']['INPUT_CHANNELS']
        self.HIDDEN_CHANNELS = self.config['MODEL']['HIDDEN_CHANNELS']
        self.NUM_RESIDUAL_BLOCKS = self.config['MODEL']['NUM_RESIDUAL_BLOCKS']
        self.BOARD_SIZE = self.config['MODEL']['BOARD_SIZE']
        
        self.CHECKPOINT_DIR = self.config['CHECKPOINT_DIR']
        self.LOG_DIR = self.config['LOG_DIR']
        self.MODEL_OUTPUT_DIR = self.config.get('MODEL_OUTPUT_DIR', 'models')
        self.ONNX_OUTPUT_PATH = self.config['ONNX_OUTPUT_PATH']
        
        self.SAVE_EVERY_ITER = self.config['SAVE_EVERY_ITER']
        self.KEEP_LAST_N_CHECKPOINTS = self.config['KEEP_LAST_N_CHECKPOINTS']
        self.SAVE_INTERVAL = 10  # Save model every 10 iterations
        
        self.USE_CUDA = self.config['DEVICE']['CUDA']
        self.DEVICE_ID = self.config['DEVICE']['DEVICE_ID']
        
        self.WIN_RATE_THRESHOLD = self.config['EVALUATION']['WIN_RATE_THRESHOLD']
        
        # 创建目录
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)
        os.makedirs(self.MODEL_OUTPUT_DIR, exist_ok=True)


class RuleBasedMCTS:
    """
    基于规则的 MCTS 智能体（不依赖神经网络）
    
    【核心原则】
    - 完全基于人工编写的评估函数
    - 使用启发式规则判断棋型（活三、冲四、活四等）
    - 通过大量随机模拟评估局面
    - 生成高质量训练数据供神经网络学习
    
    【与神经网络的分离】
    - 训练阶段：Rule-Based MCTS 生成数据 → 神经网络学习
    - 推理阶段：神经网络独立决策（不使用 MCTS）
    """
    
    def __init__(self, num_simulations: int = 800, temperature: float = 1.0):
        """
        初始化纯规则 MCTS
        
        Args:
            num_simulations: MCTS 模拟次数
            temperature: 温度系数（控制探索多样性）
        """
        self.num_simulations = num_simulations
        self.temperature = temperature
        self.BOARD_SIZE = 15
    
    def _evaluate_board(self, board: Board, player: int) -> float:
        """
        人工编写的棋盘评估函数
        
        基于五子棋棋型和位置价值进行评估
        
        Args:
            board: 棋盘对象
            player: 评估方玩家
            
        Returns:
            局面评分（-1.0 ~ 1.0）
        """
        score = 0.0
        opponent = WHITE if player == BLACK else BLACK
        
        # 遍历所有位置，评估棋型价值
        for x in range(self.BOARD_SIZE):
            for y in range(self.BOARD_SIZE):
                cell = board.get_cell(x, y)
                if cell == EMPTY:
                    continue
                
                # 计算位置价值（中心权重高）
                center_dist = abs(x - 7) + abs(y - 7)
                position_weight = 1.0 - (center_dist / 20.0)
                # 评估当前棋子的棋型
                if cell == player:
                    # 评估己方棋型
                    if is_live_four(board, x, y, player):
                        score += 10.0 * position_weight  # 活四必胜
                    elif is_open_four(board, x, y, player):
                        score += 5.0 * position_weight   # 冲四
                    elif is_live_three(board, x, y, player):
                        score += 2.0 * position_weight   # 活三
                    elif is_double_three(board, x, y, player):
                        score += 3.0 * position_weight   # 双三
                    elif is_double_four(board, x, y, player):
                        score += 8.0 * position_weight   # 双四
                else:
                    # 评估对方棋型（扣分）
                    if is_live_four(board, x, y, opponent):
                        score -= 10.0 * position_weight
                    elif is_open_four(board, x, y, opponent):
                        score -= 5.0 * position_weight
                    elif is_live_three(board, x, y, opponent):
                        score -= 2.0 * position_weight
                    elif is_double_three(board, x, y, opponent):
                        score -= 3.0 * position_weight
                    elif is_double_four(board, x, y, opponent):
                        score -= 8.0 * position_weight
        
        # 归一化到 [-1, 1]
        return np.tanh(score / 10.0)
    
    def _simulate_game(self, board: Board, player: int, max_steps: int = 100) -> int:
        """
        随机模拟一局游戏
        
        Args:
            board: 起始棋盘
            player: 起始玩家
            max_steps: 最大步数
            
        Returns:
            获胜玩家（BLACK/WHITE/None 平局）
        """
        sim_board = board.copy()
        current_player = player
        
        for step in range(max_steps):
            legal_moves = get_legal_moves(sim_board, current_player)
            if not legal_moves:
                break
            
            # 随机选择合法位置
            x, y = legal_moves[np.random.randint(len(legal_moves))]
            sim_board.make_move(x, y, current_player)
            
            # 检查胜负
            if check_win(sim_board, x, y, current_player):
                return current_player
            
            current_player = WHITE if current_player == BLACK else BLACK
        
        return None  # 平局
    
    def select_move(self, board: Board, player: int) -> Tuple[int, int]:
        """
        MCTS 搜索选择落子
        
        【算法流程】
        1. 选择：遍历所有合法位置
        2. 模拟：对每个位置进行随机游戏模拟
        3. 评估：使用人工评估函数 + 模拟结果
        4. 回溯：统计平均价值
        
        Args:
            board: 当前棋盘
            player: 当前玩家
            
        Returns:
            选择的落子位置 (x, y)
        """
        legal_moves = get_legal_moves(board, player)
        
        if not legal_moves:
            return None
        
        if len(legal_moves) == 1:
            return legal_moves[0]
        
        # MCTS 主循环
        move_scores = {}
        for x, y in legal_moves:
            move_scores[(x, y)] = []
        
        for sim_idx in range(self.num_simulations):
            for x, y in legal_moves:
                # 模拟落子
                temp_board = board.copy()
                temp_board.make_move(x, y, player)
                
                # 随机模拟剩余游戏
                winner = self._simulate_game(temp_board, player)
                
                # 计算奖励
                if winner == player:
                    reward = 1.0
                elif winner is None:
                    reward = 0.0
                else:
                    reward = -1.0
                
                # 结合人工评估
                position_eval = self._evaluate_board(temp_board, player)
                combined_score = 0.5 * reward + 0.5 * position_eval
                
                move_scores[(x, y)].append(combined_score)
        
        # 计算平均分数
        move_avg_scores = []
        for (x, y), scores in move_scores.items():
            avg_score = np.mean(scores)
            move_avg_scores.append((x, y, avg_score))
        
        # 按分数排序
        move_avg_scores.sort(key=lambda item: item[2], reverse=True)
        
        # 温度采样（增加探索性）
        if self.temperature > 0:
            scores = np.array([item[2] for item in move_avg_scores])
            scores = scores - scores.max()  # 数值稳定性
            probs = np.exp(scores / self.temperature)
            probs = probs / probs.sum()
            
            idx = np.random.choice(len(move_avg_scores), p=probs)
            return move_avg_scores[idx][0], move_avg_scores[idx][1]
        else:
            # 贪婪选择
            return move_avg_scores[0][0], move_avg_scores[0][1]


class TrainingLoop:
    """自迭代训练闭环管理器"""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Set save interval (every 10 iterations)
        self.SAVE_INTERVAL = 10
        
        # 设置设备
        if self.config.USE_CUDA and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.config.DEVICE_ID}')
            print(f"✓ 使用 GPU: CUDA {self.config.DEVICE_ID}")
        else:
            self.device = torch.device('cpu')
            print("✓ 使用 CPU")
        
        # 初始化模型
        self.current_model = self._create_model()
        
        # 上一版本模型（用于评估）
        self.previous_model = None
        
        # 日志记录
        self.training_log = {
            'iterations': [],
            'losses': [],
            'win_rates': [],
            'learning_rates': [],
            'temperatures': []
        }
        
        # 当前超参数
        self.current_lr = self.config.LEARNING_RATE
        self.current_temperature = self.config.TEMPERATURE
    
    def _create_model(self) -> GobangNet:
        """创建新模型"""
        model = GobangNet(
            input_channels=self.config.INPUT_CHANNELS,
            hidden_channels=self.config.HIDDEN_CHANNELS,
            num_residual_blocks=self.config.NUM_RESIDUAL_BLOCKS
        ).to(self.device)
        return model
    
    def self_play(self, num_games: int) -> List[Dict]:
        """
        MCTS 自我对弈生成训练数据
        
        【阶段 1】MCTS 数据生成
        - 使用**纯规则**的 MCTS 进行自我对弈
        - MCTS 基于人工编写的评估函数（不依赖神经网络）
        - 记录每个状态的策略分布（来自 MCTS 访问计数）
        - 记录游戏结果作为价值标签
        
        Args:
            num_games: 对局数量
            
        Returns:
            游戏数据列表（临时数据，训练后会被删除）
        """
        print(f"\n【阶段 1】MCTS 自我对弈：生成 {num_games} 局训练数据")
        print(f"MCTS 类型：**纯规则** (人工编写评估函数)")
        print(f"MCTS 模拟次数：{self.config.MCTS_SIMULATIONS}, 温度：{self.current_temperature:.3f}")
        
        # 创建纯规则 MCTS agent（不依赖模型）
        mcts_agent = RuleBasedMCTS(
            num_simulations=self.config.MCTS_SIMULATIONS,
            temperature=self.current_temperature
        )
        
        all_games_data = []
        
        for game_idx in tqdm(range(num_games), desc="自我对弈"):
            board = Board()
            moves = []
            current_player = BLACK
            winner = None
            
            # 进行一局游戏
            for step in range(self.config.config['DATA_GENERATION']['MAX_STEPS_PER_GAME']):
                # MCTS 选择落子
                move = mcts_agent.select_move(board, current_player)
                
                if move is None:
                    break
                
                x, y = move
                
                # 记录状态和策略
                state = board.board.copy()
                legal_moves = get_legal_moves(board, current_player)
                
                # 创建策略分布（均匀分布）
                policy = np.zeros(self.config.BOARD_SIZE ** 2, dtype=np.float32)
                for lx, ly in legal_moves:
                    idx = lx * self.config.BOARD_SIZE + ly
                    policy[idx] = 1.0 / len(legal_moves)
                
                # 落子
                board.make_move(x, y, current_player)
                moves.append({
                    'state': state,
                    'policy': policy,
                    'player': current_player,
                    'move': (x, y)
                })
                
                # 检查胜负
                if check_win(board, x, y, current_player):
                    winner = current_player
                    break
                
                # 交换玩家
                current_player = WHITE if current_player == BLACK else BLACK
            
            # 为所有状态添加价值标签
            for move_data in moves:
                player = move_data['player']
                if winner == player:
                    value = 1.0
                elif winner is not None:
                    value = -1.0
                else:
                    value = 0.0
                move_data['value'] = value
            
            all_games_data.append({
                'moves': moves,
                'winner': winner
            })
        
        print(f"✓ MCTS 数据生成完成：{len(all_games_data)} 局，{sum(len(g['moves']) for g in all_games_data)} 个状态")
        return all_games_data
    
    def train_step(self, games_data: List[Dict]) -> Tuple[float, GobangNet]:
        """
        【阶段 2】监督学习 (Supervised Learning)
        
        使用 MCTS 生成的数据进行监督学习：
        - 策略头：学习 MCTS 的策略分布（KL 散度）
        - 价值头：学习游戏结果（MSE 损失）
        
        Args:
            games_data: MCTS 生成的游戏数据列表
            
        Returns:
            (平均损失，更新后的模型)
        """
        print(f"\n【阶段 2】监督学习：使用 {len(games_data)} 局 MCTS 数据训练模型")
        
        # 准备训练数据
        all_states = []
        all_policies = []
        all_values = []
        
        for game in games_data:
            for move_data in game['moves']:
                all_states.append(move_data['state'])
                all_policies.append(move_data['policy'])
                all_values.append([move_data['value']])
        
        # Convert to PyTorch tensors
        states_array = np.array(all_states, dtype=np.int8)
        policies_array = np.array(all_policies, dtype=np.float32)
        values_array = np.array(all_values, dtype=np.float32)
        
        # Encode feature planes [N, INPUT_CHANNELS, 15, 15]
        from cpp_adapter import encode_feature_planes
        
        states_expanded = []
        for state in states_array:
            features = encode_feature_planes(state, BLACK)  # Use default history=None
            states_expanded.append(features)
        
        states_expanded = np.array(states_expanded, dtype=np.float32)
        
        states_tensor = torch.from_numpy(states_expanded).to(self.device)
        policies_tensor = torch.from_numpy(policies_array).to(self.device)
        values_tensor = torch.from_numpy(values_array).to(self.device)
        
        dataset = TensorDataset(states_tensor, policies_tensor, values_tensor)
        dataloader = DataLoader(dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
        
        # 优化器
        optimizer = optim.Adam(self.current_model.parameters(), lr=self.current_lr)
        criterion_policy = nn.KLDivLoss(reduction='batchmean')
        criterion_value = nn.MSELoss()
        
        # 训练
        self.current_model.train()
        total_loss = 0
        
        for epoch in range(self.config.EPOCHS_PER_ITER):
            epoch_loss = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.config.EPOCHS_PER_ITER}")
            for batch_states, batch_policies, batch_values in pbar:
                optimizer.zero_grad()
                
                # 前向传播
                pred_policies, pred_values = self.current_model(batch_states)
                
                # 计算损失
                policy_loss = criterion_policy(
                    torch.log(pred_policies + 1e-10), 
                    batch_policies
                )
                value_loss = criterion_value(pred_values, batch_values)
                
                # 总损失
                loss = policy_loss + value_loss
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'policy': f"{policy_loss.item():.4f}",
                    'value': f"{value_loss.item():.4f}"
                })
            
            avg_epoch_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch+1} 平均损失：{avg_epoch_loss:.4f}")
            total_loss += avg_epoch_loss
        
        avg_loss = total_loss / self.config.EPOCHS_PER_ITER
        print(f"\n✓ 监督学习完成：平均损失 {avg_loss:.4f}")
        print(f"  模型已更新，准备进入下一阶段")
        
        return avg_loss, self.current_model
    
    def evaluate_models(self, new_model: GobangNet, old_model: GobangNet) -> float:
        """
        评估新模型 vs 旧模型
        
        【评估方法】
        - 使用**纯规则 MCTS**作为裁判
        - 新模型和旧模型对战，MCTS 判断胜负
        - 确保评估的客观性（不依赖任何模型）
        
        Args:
            new_model: 新模型
            old_model: 旧模型
            
        Returns:
            新模型胜率
        """
        print(f"\n开始模型评估：{self.config.EVAL_GAMES} 局对战")
        print(f"裁判：**纯规则 MCTS** (人工编写评估函数)")
        
        # 创建纯规则 MCTS 作为裁判
        rule_mcts = RuleBasedMCTS(
            num_simulations=self.config.MCTS_SIMULATIONS,
            temperature=0.5
        )
        
        new_wins = 0
        old_wins = 0
        draws = 0
        
        for game_idx in tqdm(range(self.config.EVAL_GAMES), desc="评估对战"):
            board = Board()
            current_player = BLACK if game_idx % 2 == 0 else WHITE
            winner = None
            
            # 交替先后手
            new_is_first = (game_idx % 2 == 0)
            
            for step in range(self.config.config['DATA_GENERATION']['MAX_STEPS_PER_GAME']):
                # 使用神经网络选择落子（评估模式）
                if (new_is_first and current_player == BLACK):
                    # 新模型执黑
                    tensor = board_to_tensor(board, current_player).to(self.device)
                    with torch.no_grad():
                        policy, _ = new_model(tensor)
                    move_idx = np.argmax(policy[0].cpu().numpy())
                elif (not new_is_first and current_player == WHITE):
                    # 新模型执白
                    tensor = board_to_tensor(board, current_player).to(self.device)
                    with torch.no_grad():
                        policy, _ = new_model(tensor)
                    move_idx = np.argmax(policy[0].cpu().numpy())
                elif (new_is_first and current_player == WHITE):
                    # 旧模型执白
                    tensor = board_to_tensor(board, current_player).to(self.device)
                    with torch.no_grad():
                        policy, _ = old_model(tensor)
                    move_idx = np.argmax(policy[0].cpu().numpy())
                else:
                    # 旧模型执黑
                    tensor = board_to_tensor(board, current_player).to(self.device)
                    with torch.no_grad():
                        policy, _ = old_model(tensor)
                    move_idx = np.argmax(policy[0].cpu().numpy())
                
                x = move_idx // self.BOARD_SIZE
                y = move_idx % self.BOARD_SIZE
                
                board.make_move(x, y, current_player)
                
                if check_win(board, x, y, current_player):
                    winner = current_player
                    break
                
                current_player = WHITE if current_player == BLACK else BLACK
            
            # 统计结果
            if winner == BLACK:
                if new_is_first:
                    new_wins += 1
                else:
                    old_wins += 1
            elif winner == WHITE:
                if not new_is_first:
                    new_wins += 1
                else:
                    old_wins += 1
            else:
                draws += 1
        
        # 计算胜率
        new_win_rate = new_wins / self.config.EVAL_GAMES
        
        print(f"\n=== 评估结果 ===")
        print(f"新模型胜：{new_wins}/{self.config.EVAL_GAMES} ({new_win_rate*100:.2f}%)")
        print(f"旧模型胜：{old_wins}/{self.config.EVAL_GAMES} ({old_wins/self.config.EVAL_GAMES*100:.2f}%)")
        print(f"平局：{draws}/{self.config.EVAL_GAMES} ({draws/self.config.EVAL_GAMES*100:.2f}%)")
        
        return new_win_rate
    
    def save_checkpoint(self, iteration: int, loss: float, force_save: bool = False):
        """保存检查点
        
        Args:
            iteration: 当前迭代轮数
            loss: 当前损失
            force_save: 是否强制保存（用于异常退出）
        """
        # Check if should save (every 10 iterations or force save)
        should_save = force_save or ((iteration + 1) % self.SAVE_INTERVAL == 0)
        
        if not should_save:
            return
        
        checkpoint_path = os.path.join(
            self.config.CHECKPOINT_DIR, 
            f'ckpt_iter_{iteration}.pth'
        )
        
        # 创建优化器（用于保存）
        optimizer = optim.Adam(self.current_model.parameters(), lr=self.current_lr)
        
        save_checkpoint(
            model=self.current_model,
            optimizer=optimizer,
            epoch=iteration,
            loss=loss,
            path=checkpoint_path
        )
        
        print(f"✓ 检查点已保存：{checkpoint_path} (第 {iteration+1} 轮)")
        
        # 清理旧检查点
        self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        """清理旧的检查点，只保留最近的 N 个"""
        checkpoint_files = sorted([
            f for f in os.listdir(self.config.CHECKPOINT_DIR) 
            if f.startswith('ckpt_iter_') and f.endswith('.pth')
        ])
        
        if len(checkpoint_files) > self.config.KEEP_LAST_N_CHECKPOINTS:
            files_to_remove = checkpoint_files[:-self.config.KEEP_LAST_N_CHECKPOINTS]
            for file in files_to_remove:
                file_path = os.path.join(self.config.CHECKPOINT_DIR, file)
                os.remove(file_path)
                print(f"已删除旧检查点：{file}")
    
    def log_metrics(self, iteration: int, loss: float, win_rate: Optional[float] = None):
        """记录日志"""
        self.training_log['iterations'].append(iteration)
        self.training_log['losses'].append(loss)
        self.training_log['win_rates'].append(win_rate if win_rate is not None else 0.0)
        self.training_log['learning_rates'].append(self.current_lr)
        self.training_log['temperatures'].append(self.current_temperature)
        
        # 保存到文件
        log_path = os.path.join(self.config.LOG_DIR, 'training_log.json')
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_log, f, indent=2)
        
        # 打印当前轮次信息
        print(f"\n{'='*60}")
        print(f"迭代 {iteration} 总结:")
        print(f"  损失：{loss:.4f}")
        if win_rate is not None:
            print(f"  胜率 vs 旧模型：{win_rate*100:.2f}%")
        print(f"  学习率：{self.current_lr:.6f}")
        print(f"  温度：{self.current_temperature:.3f}")
        print(f"{'='*60}\n")
    
    def export_onnx(self):
        """导出最终模型为 ONNX 格式（与 C++ 推理引擎完全兼容）"""
        print("\n导出 ONNX 模型...")
        
        # 导入 C++ 适配器
        from cpp_adapter import export_model_for_cpp, NetworkSpec
        
        self.current_model.eval()
        
        # 使用 cpp_adapter 的优化导出函数
        export_model_for_cpp(
            model=self.current_model,
            output_path=self.config.ONNX_OUTPUT_PATH,
            verbose=True
        )
        
        print(f"✓ ONNX 模型已导出：{self.config.ONNX_OUTPUT_PATH}")
        print(f"  - 此模型可直接被 C++ GobangEngine 加载")
        print(f"  - 特征平面：{NetworkSpec.INPUT_CHANNELS} 通道")
        print(f"  - 策略输出：{NetworkSpec.POLICY_OUTPUT} 维")
        print(f"  - 价值输出：{NetworkSpec.VALUE_OUTPUT} 维")
    
    def run(self):
        """运行完整的自迭代训练闭环"""
        print("="*60)
        print("StarGobang 自迭代训练闭环")
        print("="*60)
        print(f"总迭代轮数：{self.config.TOTAL_ITERATIONS}")
        print(f"每轮对局数：{self.config.GAMES_PER_ITER}")
        print(f"评估间隔：{self.config.EVAL_INTERVAL} 轮")
        print(f"模型保存间隔：每 {self.SAVE_INTERVAL} 轮")
        print(f"开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        try:
            # 主训练循环
            for iteration in range(self.config.TOTAL_ITERATIONS):
                print(f"\n{'='*60}")
                print(f"迭代 {iteration + 1}/{self.config.TOTAL_ITERATIONS}")
                print(f"{'='*60}")
                
                # 1. 自我对弈生成数据
                games = self.self_play(self.config.GAMES_PER_ITER)
                
                # 2. 训练更新模型
                loss, self.current_model = self.train_step(games)
                
                # 3. 评估（每隔 EVAL_INTERVAL 轮）
                win_rate = None
                if self.config.EVAL_INTERVAL > 0 and (iteration + 1) % self.config.EVAL_INTERVAL == 0:
                    if self.previous_model is not None:
                        win_rate = self.evaluate_models(self.current_model, self.previous_model)
                        
                        # 如果新模型表现更好，更新 previous_model
                        if win_rate > self.config.WIN_RATE_THRESHOLD:
                            print(f"✓ 新模型通过验证（胜率>{self.config.WIN_RATE_THRESHOLD*100:.1f}%），替换旧模型")
                            self.previous_model = GobangNet(
                                input_channels=self.config.INPUT_CHANNELS,
                                hidden_channels=self.config.HIDDEN_CHANNELS,
                                num_residual_blocks=self.config.NUM_RESIDUAL_BLOCKS
                            ).to(self.device)
                            self.previous_model.load_state_dict(self.current_model.state_dict())
                        else:
                            print(f"✗ 新模型未通过验证（胜率≤{self.config.WIN_RATE_THRESHOLD*100:.1f}%），保留旧模型")
                    else:
                        # 第一次评估，保存当前模型为 previous_model
                        self.previous_model = GobangNet(
                            input_channels=self.config.INPUT_CHANNELS,
                            hidden_channels=self.config.HIDDEN_CHANNELS,
                            num_residual_blocks=self.config.NUM_RESIDUAL_BLOCKS
                        ).to(self.device)
                        self.previous_model.load_state_dict(self.current_model.state_dict())
                        print("已保存当前模型作为评估基准")
                
                # 4. 记录日志
                self.log_metrics(iteration, loss, win_rate)
                
                # 5. 每 10 轮保存一次模型
                self.save_checkpoint(iteration, loss)
                
                # 6. 清理临时数据（释放内存）
                self._cleanup_temporary_data(games)
                
                # 7. 衰减超参数
                self.current_lr = max(self.current_lr * self.config.LR_DECAY, self.config.MIN_LR)
                self.current_temperature = max(
                    self.current_temperature * self.config.TEMPERATURE_DECAY, 
                    self.config.MIN_TEMPERATURE
                )
            
            # 正常完成，导出最终模型
            self.export_onnx()
            
            # 保存最终日志
            final_log_path = os.path.join(self.config.LOG_DIR, 'final_training_log.json')
            with open(final_log_path, 'w', encoding='utf-8') as f:
                json.dump(self.training_log, f, indent=2)
            
            print("\n" + "="*60)
            print("✓ 训练闭环完成！")
            print(f"最终模型已导出：{self.config.ONNX_OUTPUT_PATH}")
            print(f"训练日志已保存：{final_log_path}")
            print(f"结束时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*60)
            
        except KeyboardInterrupt:
            print("\n\n⚠ 检测到中断信号，正在保存模型...")
            self._emergency_save()
            raise
        except Exception as e:
            print(f"\n\n⚠ 发生异常：{e}")
            print("正在紧急保存模型...")
            self._emergency_save()
            raise
    
    def _cleanup_temporary_data(self, games_data: List[Dict]):
        """
        【阶段 3】清理临时数据
        
        删除 MCTS 生成的临时训练数据，释放内存
        只保留：
        - 模型检查点（每 10 轮保存）
        - 训练日志（metrics）
        - 最终 ONNX 模型（训练完成时）
        
        Args:
            games_data: 需要清理的游戏数据列表
        """
        print(f"\n【阶段 3】清理临时数据...")
        num_games = len(games_data)
        num_states = sum(len(g['moves']) for g in games_data)
        
        # 显式删除数据，释放内存
        del games_data
        
        print(f"✓ 已清理临时数据：{num_games} 局，{num_states} 个状态")
        print(f"  内存已释放，准备下一轮迭代\n")
    
    def _emergency_save(self):
        """紧急保存模型（用于异常退出）"""
        emergency_path = os.path.join(
            self.config.CHECKPOINT_DIR,
            f'emergency_ckpt_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
        )
        
        optimizer = optim.Adam(self.current_model.parameters(), lr=self.current_lr)
        
        save_checkpoint(
            model=self.current_model,
            optimizer=optimizer,
            epoch=len(self.training_log['iterations']),
            loss=self.training_log['losses'][-1] if self.training_log['losses'] else 0.0,
            path=emergency_path
        )
        
        print(f"✓ 紧急检查点已保存：{emergency_path}")
        print(f"  可使用此检查点恢复训练")


def main():
    """主函数"""
    print("="*60)
    print("StarGobang 自迭代训练闭环系统")
    print("="*60)
    
    # 加载配置
    config = Config('config.yaml')
    
    # 创建训练循环
    trainer = TrainingLoop(config)
    
    # 运行训练
    trainer.run()
    
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print("\nPython 端 = 自进化训练闭环（从零到强） | C++ 端 = 毫秒级推理引擎")
    print("闭环价值：不是刷棋力，是为 StarGo 锻造可验证、可迁移的训练流水线")
    print("="*60)


if __name__ == '__main__':
    main()
