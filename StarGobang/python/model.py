"""
五子棋神经网络模型

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
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, channels: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += identity
        out = F.relu(out)
        
        return out


class GobangNet(nn.Module):
    """
    五子棋神经网络
    
    输入：15x15x3 (黑棋位置，白棋位置，当前玩家)
    输出：
      - policy: 225 维概率分布（下一步落子位置）
      - value: [-1, 1] 标量（当前局面评估值）
    """
    
    def __init__(self, input_channels: int = 3, hidden_channels: int = 256, 
                 num_residual_blocks: int = 10):
        super().__init__()
        
        # 初始卷积层
        self.conv_input = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(hidden_channels)
        
        # 残差塔
        self.residual_tower = nn.Sequential(*[
            ResidualBlock(hidden_channels) for _ in range(num_residual_blocks)
        ])
        
        # 策略头（Policy Head）- 输出 225 个位置的落子概率
        self.policy_conv = nn.Conv2d(hidden_channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 15 * 15, 225)
        
        # 价值头（Value Head）- 输出局面评估值 [-1, 1]
        self.value_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(15 * 15, 128)
        self.value_fc2 = nn.Linear(128, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, 3, 15, 15]
            
        Returns:
            policy: [batch_size, 225] 落子概率分布
            value: [batch_size, 1] 局面评估值
        """
        # 初始卷积
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # 残差网络
        x = self.residual_tower(x)
        
        # 策略头
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        policy = F.softmax(policy, dim=-1)
        
        # 价值头
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value
    
    def get_policy(self, x: torch.Tensor) -> torch.Tensor:
        """仅获取策略输出"""
        policy, _ = self.forward(x)
        return policy
    
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """仅获取价值输出"""
        _, value = self.forward(x)
        return value


def create_model(pretrained_path: str = None) -> GobangNet:
    """
    创建模型并可选择加载预训练权重
    
    Args:
        pretrained_path: 预训练模型路径（可选）
        
    Returns:
        模型实例
    """
    model = GobangNet()
    
    if pretrained_path:
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"已加载预训练模型：{pretrained_path}")
    
    return model


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                    epoch: int, loss: float, path: str):
    """
    保存训练检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前轮次
        loss: 损失值
        path: 保存路径
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)
    print(f"检查点已保存：{path}")


def load_checkpoint(path: str, model: nn.Module, 
                    optimizer: torch.optim.Optimizer = None) -> dict:
    """
    加载训练检查点
    
    Args:
        path: 检查点路径
        model: 模型
        optimizer: 优化器（可选）
        
    Returns:
        检查点信息
    """
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"检查点已加载：{path} (epoch={checkpoint['epoch']}, loss={checkpoint['loss']:.4f})")
    return checkpoint


def board_to_tensor(board: 'game.Board', current_player: int, 
                    history=None) -> torch.Tensor:
    """
    Convert board to model input tensor (aligned with C++ Board::create_feature_tensor)
    
    Args:
        board: Board object
        current_player: Current player
        history: History moves [(x, y, player), ...]
        
    Returns:
        Input tensor [1, INPUT_CHANNELS, 15, 15]
        
    [Compatibility Note]
      - Uses 10-channel feature encoding (consistent with C++)
    """
    import numpy as np
    from game import BLACK, WHITE
    
    # Use C++ compatible feature encoding
    try:
        from cpp_adapter import encode_feature_planes, NetworkSpec
        
        # Use C++ compatible feature encoding
        features = encode_feature_planes(
            board.board if hasattr(board, 'board') else board,
            current_player,
            history
        )
        
        # 添加 batch 维度 [1, INPUT_CHANNELS, 15, 15]
        tensor = torch.from_numpy(features).unsqueeze(0)
        return tensor
        
    except ImportError:
        # Fallback to basic 3-channel mode (backward compatibility)
        black_plane = (board.board == BLACK).astype(np.float32)
        white_plane = (board.board == WHITE).astype(np.float32)
        
        # Current player channel (all 1 for BLACK turn, all 0 for WHITE turn)
        player_plane = np.full((15, 15), current_player == BLACK, dtype=np.float32)
        
        # Stack as [3, 15, 15]
        tensor = np.stack([black_plane, white_plane, player_plane], axis=0)
        
        # Add batch dimension [1, 3, 15, 15]
        tensor = torch.from_numpy(tensor).unsqueeze(0)
        
        return tensor


def batch_boards_to_tensor(boards: list, current_players: list) -> torch.Tensor:
    """
    批量将棋盘转换为模型输入张量
    
    Args:
        boards: 棋盘列表
        current_players: 当前玩家列表
        
    Returns:
        输入张量 [batch_size, 3, 15, 15]
    """
    tensors = []
    for board, player in zip(boards, current_players):
        tensors.append(board_to_tensor(board, player))
    return torch.cat(tensors, dim=0)
