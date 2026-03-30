#!/bin/bash
# 模型文件管理脚本 - 在 Python 训练闭环和 C++ 推理引擎之间同步模型
# MIT License
# Copyright (c) 2026 StarGobang Team

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_MODELS_DIR="${SCRIPT_DIR}/../python/models"
CPP_MODELS_DIR="${SCRIPT_DIR}/models"

echo "=========================================="
echo "  StarGobang 模型文件管理工具"
echo "=========================================="
echo ""

# 检查参数
if [ $# -lt 1 ]; then
    echo "用法："
    echo "  $0 <操作> [参数]"
    echo ""
    echo "操作:"
    echo "  copy-to-cpp [模型名]  - 将 Python 训练的模型复制到 C++ 目录"
    echo "  list                  - 列出所有可用模型"
    echo "  info                  - 显示当前模型配置"
    echo ""
    echo "示例:"
    echo "  $0 list                           # 查看所有可用模型"
    echo "  $0 copy-to-cpp model_gobang.onnx  # 复制五子棋模型"
    echo ""
    exit 1
fi

ACTION="$1"

case "$ACTION" in
    list)
        echo "Python 训练闭环中的模型:"
        echo "------------------------"
        if [ -d "$PYTHON_MODELS_DIR" ]; then
            ls -lh "$PYTHON_MODELS_DIR"/*.onnx 2>/dev/null || echo "  (无 ONNX 模型文件)"
        else
            echo "  (模型目录不存在)"
        fi
        echo ""
        
        echo "C++ 推理引擎中的模型:"
        echo "----------------------"
        if [ -d "$CPP_MODELS_DIR" ]; then
            ls -lh "$CPP_MODELS_DIR"/*.onnx 2>/dev/null || echo "  (无 ONNX 模型文件)"
        else
            echo "  (模型目录不存在)"
        fi
        ;;
        
    copy-to-cpp)
        if [ $# -lt 2 ]; then
            echo "错误：请指定模型文件名"
            echo "示例：$0 copy-to-cpp model_gobang.onnx"
            exit 1
        fi
        
        MODEL_NAME="$2"
        SOURCE="${PYTHON_MODELS_DIR}/${MODEL_NAME}"
        TARGET="${CPP_MODELS_DIR}/${MODEL_NAME}"
        
        echo "步骤 1/3: 检查源模型文件..."
        if [ ! -f "$SOURCE" ]; then
            echo "❌ 错误：模型文件不存在：$SOURCE"
            exit 1
        fi
        echo "✓ 找到模型：$SOURCE"
        echo ""
        
        echo "步骤 2/3: 创建目标目录..."
        mkdir -p "$CPP_MODELS_DIR"
        echo "✓ 目标目录：$CPP_MODELS_DIR"
        echo ""
        
        echo "步骤 3/3: 复制模型文件..."
        cp "$SOURCE" "$TARGET"
        echo "✓ 已复制：$MODEL_NAME"
        echo "  大小：$(ls -lh "$TARGET" | awk '{print $5}')"
        echo ""
        
        echo "=========================================="
        echo "  ✅ 模型复制完成！"
        echo "=========================================="
        echo ""
        echo "下一步操作:"
        echo "  1. 编译 C++ 项目:"
        echo "     cd cmake-build-release && ninja"
        echo ""
        echo "  2. 运行推理引擎:"
        echo "     ./gobang_demo models/${MODEL_NAME}"
        echo ""
        ;;
        
    info)
        echo "当前模型配置:"
        echo "============="
        echo ""
        echo "Python 训练输出目录:"
        echo "  ${PYTHON_MODELS_DIR}"
        echo ""
        echo "C++ 推理输入目录:"
        echo "  ${CPP_MODELS_DIR}"
        echo ""
        echo "符号链接状态:"
        if [ -L "$CPP_MODELS_DIR" ]; then
            echo "  ✓ 已创建符号链接"
            ls -l "$CPP_MODELS_DIR"
        else
            echo "  ✗ 未创建符号链接"
        fi
        echo ""
        
        echo "默认模型文件:"
        echo "  - model_gobang.onnx (五子棋，15x15)"
        echo ""
        ;;
        
    *)
        echo "未知操作：$ACTION"
        echo "使用 '$0' 查看用法说明"
        exit 1
        ;;
esac
