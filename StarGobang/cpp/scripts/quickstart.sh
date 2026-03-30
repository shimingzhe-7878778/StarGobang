#!/bin/bash
# 快速入门脚本 - 从训练到推理的完整流程
# MIT License
# Copyright (c) 2026 StarGobang Team

set -e

echo "╔══════════════════════════════════════════════╗"
echo "║  五子棋 AI - 从训练到推理                    ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_MODELS_DIR="${SCRIPT_DIR}/../python/models"
CPP_DIR="${SCRIPT_DIR}"

# 步骤 1: 检查 Python 训练模型
echo "步骤 1/4: 检查 Python 训练闭环..."
if [ ! -d "$PYTHON_MODELS_DIR" ]; then
    echo "❌ Python 模型目录不存在：$PYTHON_MODELS_DIR"
    echo ""
    echo "请先完成 Python 端训练："
    echo "  cd ../python"
    echo "  python train_loop.py"
    exit 1
fi

MODEL_COUNT=$(find "$PYTHON_MODELS_DIR" -name "*.onnx" -type f 2>/dev/null | wc -l)
if [ "$MODEL_COUNT" -eq 0 ]; then
    echo "❌ Python 模型目录中没有 ONNX 模型文件"
    echo ""
    echo "请先完成 Python 端训练，导出 model_gobang.onnx"
    exit 1
fi

echo "✓ 找到 $MODEL_COUNT 个 ONNX 模型文件"
ls -lh "$PYTHON_MODELS_DIR"/*.onnx 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""

# 步骤 2: 选择模型
echo "步骤 2/4: 选择要使用的模型"
echo "----------------------------------------"
select MODEL_FILE in $(ls "$PYTHON_MODELS_DIR"/*.onnx 2>/dev/null); do
    if [ -n "$MODEL_FILE" ]; then
        MODEL_NAME=$(basename "$MODEL_FILE")
        echo "✓ 已选择：$MODEL_NAME"
        break
    else
        echo "请选择有效的模型文件编号"
    fi
done
echo ""

# 步骤 3: 复制模型到 C++ 目录
echo "步骤 3/4: 复制模型到 C++ 推理引擎..."
"${SCRIPT_DIR}/manage_models.sh" copy-to-cpp "$MODEL_NAME"
echo ""

# 步骤 4: 编译并运行
echo "步骤 4/4: 编译 C++ 推理引擎..."
if [ ! -d "${CPP_DIR}/cmake-build-release" ]; then
    echo "⚠️  cmake-build-release 目录不存在，正在创建..."
    mkdir -p "${CPP_DIR}/cmake-build-release"
    cd "${CPP_DIR}/cmake-build-release"
    cmake .. -DCMAKE_BUILD_TYPE=Release
fi

cd "${CPP_DIR}/cmake-build-release"
ninja

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  ✅ 准备完成！                               ║"
echo "╠══════════════════════════════════════════════╣"
echo "║ 运行推理引擎：                              ║"
echo "║   cd cmake-build-release                     ║"
echo "║   ./gobang_demo ../models/${MODEL_NAME}      ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# 询问是否立即运行
read -p "是否立即运行推理引擎？(y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "启动推理引擎..."
    echo ""
    ./gobang_demo "../models/${MODEL_NAME}"
fi
