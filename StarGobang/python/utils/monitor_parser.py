"""
C++ Learning Monitor 解析器
解析 C++ learning_monitor 导出的 JSON 学习包

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
"""
import json
import hashlib
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class LearningPackageParser:
    """C++ 学习包解析器"""
    
    def __init__(self, json_path: str):
        """
        初始化解析器
        
        Args:
            json_path: JSON 文件路径
        """
        self.json_path = Path(json_path)
        self.data = None
        self.metadata = {}
        
        if not self.json_path.exists():
            raise FileNotFoundError(f"学习包文件不存在：{json_path}")
        
        self._load_and_validate()
    
    def _load_and_validate(self):
        """加载并验证 JSON 文件"""
        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # 验证基本结构
        required_fields = ['metadata', 'training_stats', 'model_snapshots']
        for field in required_fields:
            if field not in self.data:
                raise ValueError(f"缺少必需字段：{field}")
        
        # 提取元数据
        self.metadata = self.data.get('metadata', {})
        
        # 验证校验码
        self._verify_checksum()
    
    def _verify_checksum(self) -> bool:
        """
        验证文件完整性（校验码验证）
        
        Returns:
            True: 校验通过
            False: 校验失败
        """
        expected_checksum = self.metadata.get('checksum')
        if not expected_checksum:
            print("警告：未找到校验码，跳过完整性验证")
            return True
        
        # 计算实际校验码
        with open(self.json_path, 'rb') as f:
            content = f.read()
        
        # 临时移除 checksum 字段后重新计算
        data_copy = self.data.copy()
        if 'checksum' in data_copy.get('metadata', {}):
            del data_copy['metadata']['checksum']
        
        content_without_checksum = json.dumps(data_copy, sort_keys=True).encode('utf-8')
        actual_checksum = hashlib.sha256(content_without_checksum).hexdigest()
        
        if actual_checksum != expected_checksum:
            print(f"警告：校验码不匹配")
            print(f"预期：{expected_checksum}")
            print(f"实际：{actual_checksum}")
            return False
        
        print("校验码验证通过 ✓")
        return True
    
    def get_distribution_shift(self) -> float:
        """
        计算分布偏移（KL 散度）
        
        Returns:
            KL 散度值
        """
        old_dist = self.data.get('policy_distribution', {}).get('old', [])
        new_dist = self.data.get('policy_distribution', {}).get('new', [])
        
        if not old_dist or not new_dist:
            return 0.0
        
        old_dist = np.array(old_dist, dtype=np.float64)
        new_dist = np.array(new_dist, dtype=np.float64)
        
        # 归一化
        old_dist = old_dist / (old_dist.sum() + 1e-10)
        new_dist = new_dist / (new_dist.sum() + 1e-10)
        
        # 计算 KL 散度：D_KL(P||Q) = Σ P(i) log(P(i)/Q(i))
        kl_divergence = np.sum(old_dist * np.log((old_dist + 1e-10) / (new_dist + 1e-10)))
        
        return float(kl_divergence)
    
    def get_win_rate_fluctuation(self) -> Dict[str, float]:
        """
        获取胜率波动特征
        
        Returns:
            胜统计字典
        """
        stats = self.data.get('training_stats', {})
        
        return {
            'old_win_rate': stats.get('old_model_win_rate', 0.5),
            'new_win_rate': stats.get('new_model_win_rate', 0.5),
            'win_rate_change': stats.get('new_model_win_rate', 0.5) - 
                              stats.get('old_model_win_rate', 0.5),
            'total_games': stats.get('total_games', 0),
            'avg_game_length': stats.get('avg_game_length', 0)
        }
    
    def get_model_snapshots(self) -> List[Dict]:
        """
        获取模型快照列表
        
        Returns:
            模型快照列表
        """
        return self.data.get('model_snapshots', [])
    
    def get_training_features(self) -> Dict:
        """
        提取训练特征
        
        Returns:
            特征字典
        """
        features = {
            'kl_divergence': self.get_distribution_shift(),
            'win_rate_stats': self.get_win_rate_fluctuation(),
            'learning_rate': self.metadata.get('learning_rate', 1e-4),
            'temperature': self.metadata.get('temperature', 1.0),
            'mcts_simulations': self.metadata.get('mcts_simulations', 800)
        }
        
        return features
    
    def export_for_finetuning(self, output_path: str):
        """
        导出用于微调的数据格式
        
        Args:
            output_path: 输出文件路径
        """
        export_data = {
            'features': self.get_training_features(),
            'snapshots': self.get_model_snapshots(),
            'recommendation': self._get_finetuning_recommendation()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"微调数据已导出：{output_path}")
    
    def _get_finetuning_recommendation(self) -> Dict:
        """
        获取微调建议
        
        Returns:
            建议字典
        """
        kl = self.get_distribution_shift()
        win_stats = self.get_win_rate_fluctuation()
        
        recommendation = {
            'should_finetune': False,
            'reason': '',
            'suggested_layers': [],
            'suggested_lr': 1e-5,
            'suggested_epochs': 3
        }
        
        # 如果分布偏移较大或胜率下降，建议微调
        if kl > 0.1:
            recommendation['should_finetune'] = True
            recommendation['reason'] = f'检测到显著分布偏移 (KL={kl:.4f})'
            recommendation['suggested_layers'] = ['policy_fc', 'value_fc2']
        
        elif win_stats['win_rate_change'] < -0.05:
            recommendation['should_finetune'] = True
            recommendation['reason'] = f'胜率下降 {abs(win_stats["win_rate_change"])*100:.2f}%'
            recommendation['suggested_layers'] = ['value_head']
        
        else:
            recommendation['reason'] = '模型状态良好，无需微调'
        
        return recommendation


def load_learning_package(json_path: str) -> Dict:
    """
    【便捷函数】加载学习包并返回解析后的数据
    
    Args:
        json_path: JSON 文件路径
        
    Returns:
        解析后的数据字典
    """
    parser = LearningPackageParser(json_path)
    
    return {
        'parser': parser,
        'features': parser.get_training_features(),
        'recommendation': parser._get_finetuning_recommendation(),
        'metadata': parser.metadata
    }


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("用法：python monitor_parser.py <learning_package.json>")
        sys.exit(1)
    
    json_path = sys.argv[1]
    
    try:
        result = load_learning_package(json_path)
        
        print("\n=== 学习包分析 ===")
        print(f"KL 散度：{result['features']['kl_divergence']:.6f}")
        print(f"胜率变化：{result['features']['win_rate_stats']['win_rate_change']*100:.2f}%")
        print(f"\n微调建议：")
        rec = result['recommendation']
        print(f"是否微调：{'是' if rec['should_finetune'] else '否'}")
        print(f"原因：{rec['reason']}")
        if rec['should_finetune']:
            print(f"建议层：{rec['suggested_layers']}")
            print(f"建议学习率：{rec['suggested_lr']}")
            print(f"建议轮次：{rec['suggested_epochs']}")
        
    except Exception as e:
        print(f"错误：{e}")
        sys.exit(1)
