#!/bin/bash
# StarGo 迁移脚本 - 从五子棋到围棋的快速切换
# 
# 【关键强化】
# - 模型来源：Python 端 train_loop.py 闭环训练最终导出的 model_go.onnx
# - 训练闭环验证了模型有效性，本引擎专注极致推理性能
# - [StarGo 迁移] 闭环训练出的围棋模型 直接替换本引擎模型文件 即得 StarGo 推理核心
#
# MIT License
# Copyright (c) 2026 StarGobang Team
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

set -e  # 遇到错误立即退出

echo "╔══════════════════════════════════════════════╗"
echo "║  StarGo 迁移脚本 - 从五子棋到围棋            ║"
echo "╠══════════════════════════════════════════════╣"
echo "║【关键强化】                                ║"
echo "║  • 模型来源：Python 端 train_loop.py 闭环训练  ║"
echo "║  • 训练闭环验证模型有效性，专注极致推理性能  ║"
echo "║  • 与 Python 训练闭环零耦合，仅依赖模型文件    ║"
echo "║  • [StarGo 迁移] 替换模型文件即得 StarGo 核心  ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# 检查参数
if [ $# -lt 1 ]; then
    echo "用法：$0 <围棋模型路径> [棋盘大小]"
    echo ""
    echo "示例:"
    echo "  $0 /path/to/model_go.onnx        # 使用默认 15x15 棋盘"
    echo "  $0 /path/to/model_go.onnx 19     # 使用 19x19 棋盘"
    echo ""
    exit 1
fi

GO_MODEL_PATH="$1"
BOARD_SIZE="${2:-15}"  # 默认 15x15

echo "步骤 1/5: 检查围棋模型文件..."
if [ ! -f "$GO_MODEL_PATH" ]; then
    echo "❌ 错误：模型文件不存在：$GO_MODEL_PATH"
    exit 1
fi
echo "✓ 模型文件存在：$GO_MODEL_PATH"
echo ""

echo "步骤 2/5: 备份当前五子棋模型..."
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

if [ -f "model.onnx" ]; then
    cp model.onnx "$BACKUP_DIR/"
    echo "✓ 已备份到：$BACKUP_DIR/model.onnx"
else
    echo "ℹ 未找到当前模型，跳过备份"
fi
echo ""

echo "步骤 3/5: 复制围棋模型..."
cp "$GO_MODEL_PATH" model.onnx
echo "✓ 已复制：$GO_MODEL_PATH -> model.onnx"
echo ""

echo "步骤 4/5: 调整网络规格（如需要）..."
if [ "$BOARD_SIZE" -eq 19 ]; then
    echo "⚙  配置 19x19 棋盘..."
    
    # 备份原始配置文件
    cp include/config.h "$BACKUP_DIR/config.h.backup"
    
    # 修改 config.h 中的网络规格
    sed -i 's/static constexpr int WIDTH = 15;/static constexpr int WIDTH = 19;/' include/config.h
    sed -i 's/static constexpr int HEIGHT = 15;/static constexpr int HEIGHT = 19;/' include/config.h
    sed -i 's/static constexpr int POLICY_OUTPUT = 225;/static constexpr int POLICY_OUTPUT = 361;/' include/config.h
    
    echo "✓ 已修改 include/config.h:"
    echo "    WIDTH = 19, HEIGHT = 19"
    echo "    POLICY_OUTPUT = 361"
    echo ""
    echo "⚠️  注意：还需要手动修改以下常量："
    echo "   - NetworkSpec::INPUT_CHANNELS (根据特征数量)"
    echo "   - 其他与棋盘大小相关的参数"
else
    echo "ℹ  使用默认 15x15 棋盘，无需修改配置"
fi
echo ""

echo "步骤 5/5: 重新编译..."
read -p "是否需要重新编译？(y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -d "build" ]; then
        cd build
        make clean
        make -j$(nproc)
        cd ..
        echo "✓ 编译完成"
    else
        echo "❌ 错误：找不到 build 目录"
        exit 1
    fi
else
    echo "ℹ  跳过编译步骤"
fi
echo ""

echo "╔══════════════════════════════════════════════╗"
echo "║  StarGo 迁移完成！                          ║"
echo "╠══════════════════════════════════════════════╣"
echo "║ 运行围棋 AI:                                 ║"
echo "║   ./build/gobang_demo model.onnx            ║"
echo "╠══════════════════════════════════════════════╣"
echo "║ 如需恢复五子棋模式：                         ║"
echo "║   cp $BACKUP_DIR/model.onnx model.onnx       ║"
echo "║   cp $BACKUP_DIR/config.h.backup include/config.h  ║"
echo "║   cd build && make clean && make             ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

echo "📊 模型信息:"
echo "  源文件：$GO_MODEL_PATH"
echo "  目标文件：./model.onnx"
echo "  棋盘大小：${BOARD_SIZE}x${BOARD_SIZE}"
echo "  备份目录：$BACKUP_DIR"
echo ""

echo "✅ StarGo 推理核心已就绪！"
echo "   模型来源：Python 端 train_loop.py 闭环训练最终导出"
echo "   训练闭环验证了模型有效性，本引擎专注极致推理性能"
