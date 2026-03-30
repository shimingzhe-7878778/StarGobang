# StarGobang C++ 纯推理引擎

五子棋 AI 纯推理引擎，基于 Python 训练闭环导出的 ONNX 模型，专注极致推理性能。

## 核心特性

- **纯推理架构**: 与 Python 训练闭环零耦合，仅依赖 `model_gobang.onnx` 文件
- **极致性能**: O3 优化 + LTO + march=native + HPC 调度
- **元学习增强**: 动态调整 MCTS、ONNX 线程、内存池等参数
- **并行搜索**: 多线程 MCTS，充分利用多核 CPU
- **MIT 许可证**: 完全开源，可商用

## 快速开始

### 编译

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### 运行

```bash
# 从 Python 训练闭环复制模型到 C++ 目录
./scripts/manage_models.sh copy-to-cpp model_gobang.onnx

# 使用模型文件运行
./gobang_demo models/model_gobang.onnx
```

**模型文件位置**:
- Python 训练输出：`StarGobang/python/models/`
- C++ 推理输入：`cpp/models/` (符号链接)
- 使用脚本管理：`./scripts/manage_models.sh`

## 项目结构

```
cpp/
├── CMakeLists.txt          # CMake 配置
├── LICENSE                 # MIT License
├── include/                # 头文件
│   ├── board.h
│   ├── config.h
│   ├── forbidden_move_detector.h
│   ├── gobang_engine.h
│   ├── hpc_scheduler.h
│   ├── learning_monitor.h
│   ├── mcts.h
│   └── meta_learner.h
├── src/                    # 源文件
│   ├── board.cpp
│   ├── forbidden_move_detector.cpp
│   ├── gobang_engine.cpp
│   ├── hpc_scheduler.cpp
│   ├── inference_engine.cpp
│   ├── learning_monitor.cpp
│   ├── main.cpp
│   ├── mcts.cpp
│   └── meta_learner.cpp
├── cmake/
│   └── FindONNXRuntime.cmake
└── scripts/
    └── migrate_to_stargo.sh
```

## 技术栈

- **语言**: C++20
- **推理后端**: ONNX Runtime (MIT)
- **构建系统**: CMake
- **优化**: -O3 -march=native -LTO
- **平台**: Ubuntu 22.04+

## 编译模式

| 模式 | 用途 | 编译命令 |
|------|------|----------|
| **Debug** | 调试、训练 | `cmake .. -DCMAKE_BUILD_TYPE=Debug` |
| **Release** | 比赛、推理 | `cmake .. -DCMAKE_BUILD_TYPE=Release` |

## StarGo 迁移

从五子棋迁移到围棋：

```bash
# 1. 准备 Python 训练闭环导出的 model_go.onnx
# 2. 运行迁移脚本
./scripts/migrate_to_stargo.sh /path/to/model_go.onnx 19

# 3. 编译运行
cd build && make
./gobang_demo model.onnx
```

## 核心组件

### GobangEngine
推理引擎核心，加载 ONNX 模型并执行推理。

### MetaLearner
元学习器，动态调整所有影响性能的参数：
- MCTS 模拟次数、并行线程数
- ONNX 内部/跨操作线程数
- 内存池大小、CPU 亲和性

### HPCScheduler
高性能计算调度器：
- CPU 亲和性绑定
- 内存池管理
- 性能分析

### MCTS
蒙特卡洛树搜索：
- 并行搜索
- PUCT 算法
- Dirichlet 噪声

## 性能指标

| 指标 | Debug | Release | 提升 |
|------|-------|---------|------|
| 文件大小 | ~6MB | ~1MB | 6x |
| 推理速度 | 基准 | 3-5x | - |
| ELO 等级分 | - | +200-400 | - |

## 依赖

- **ONNX Runtime**: v1.16.3+ (已包含在 onnxruntime/ 目录)
- **编译器**: GCC 11+ 或 Clang 13+
- **CMake**: 3.15+
