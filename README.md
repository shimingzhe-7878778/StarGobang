# StarGobang AI Training & Evaluation System

## Overview

Complete Gobang (Five-in-a-Row) AI training and evaluation system featuring **self-iterative training loop** with dual-domain architecture:

- **Python Side**: Self-evolving training loop (from zero to strong)
- **C++ Side**: Millisecond-level inference engine

**Important**: This project currently **only supports Linux** (tested on Ubuntu 24.04).

### Core Mission

> **Loop Value**: Not for grinding Elo, but to forge a verifiable, transferable training pipeline for **StarGo** (our upcoming Go AI system).

---

## Important Notices

### 🎯 **Purpose: Algorithm Verification Only**
This project is designed **solely for algorithm verification and research purposes**. It is **NOT intended for production-level deployment** or commercial use.

### ⚖️ **License: MIT**
All code in this repository is released under the **MIT License**, ensuring maximum freedom for research, experimentation, and derivative works.

### 🚀 **Future Direction: Transition to Go (StarGo)**
This gobang implementation serves as a **stepping stone and testing ground** for our ultimate goal: developing **StarGo** - a full-scale Go AI system. The architectures, training methodologies, and verification frameworks developed here will directly inform and validate the approaches used in StarGo, where true strategic depth will be realized.

---

## Quick Start

### 1. Install Dependencies

```bash
cd python
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Training Loop (Optional)

```bash
python3 train_loop.py
```

**Note**: Training takes 45-60 minutes for 50 iterations. You can skip this and use community-contributed models instead.

### 3. Build C++ Inference Engine

```bash
cd ../cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### 4. Interactive CLI (Play with AI!)

```bash
cd ../python
# Play against trained model
python3 cli.py play --model checkpoints/ckpt_iter_49.pth

# Watch AI vs AI battle
python3 cli.py watch --model1 model1.pth --model2 model2.pth --games 3

# View model information
python3 cli.py info --model checkpoints/model.pth
```

See [`python/CLI_GUIDE.md`](python/CLI_GUIDE.md) for detailed instructions.

---

## Directory Structure

```
StarGobang/
├── python/                           # Training domain
│   ├── game.py                       # Game logic & rules
│   ├── model.py                      # PyTorch model definition
│   ├── train_loop.py                 # Self-iterative training loop
│   ├── cpp_adapter.py                # Python-C++ interface adapter
│   ├── cli.py                        # Command-line interface
│   ├── eval_model.py                 # Model evaluation tool
│   ├── verify_compatibility.py       # Compatibility verification
│   ├── config.yaml                   # Training configuration
│   ├── utils/                        # Utility modules
│   │   ├── data_generator.py
│   │   └── monitor_parser.py
│   ├── checkpoints/                  # Model checkpoints
│   ├── logs/                         # Training logs
│   └── docs/                         # Python documentation
│       ├── README.md
│       ├── CLI_GUIDE.md
│       ├── MODEL_CONTRIBUTION_GUIDE.md
│       └── MODEL_README_TEMPLATE.md
│
├── cpp/                              # Inference domain
│   ├── include/                      # Header files
│   │   ├── board.h
│   │   ├── mcts.h
│   │   ├── gobang_engine.h
│   │   ├── meta_learner.h
│   │   └── hpc_scheduler.h
│   ├── src/                          # Source files
│   │   ├── board.cpp
│   │   ├── mcts.cpp
│   │   ├── gobang_engine.cpp
│   │   ├── meta_learner.cpp
│   │   └── hpc_scheduler.cpp
│   ├── cmake/                        # CMake modules
│   ├── scripts/                      # Utility scripts
│   ├── build/                        # Build directory
│   └── models/                       # Model files (symlink to python/models/)
│
├── README.md                         # This file (root documentation)
├── LICENSE                           # MIT License
└── .gitignore
```

---

## Core Features

### Self-Iterative Training Loop (Python)

**File**: [`python/train_loop.py`](python/train_loop.py) + [`python/config.yaml`](python/config.yaml)

**Three Elements of the Loop**:
1. **Self-play data generation** - Rule-Based MCTS self-play using current model
2. **Train new model** - Update based on generated data (KL divergence + MSE loss)
3. **Replace old model** - Replace if win rate > 55% (loop N iterations)

**Per iteration auto**:
- ✅ Save checkpoint (`ckpt_iter_X.pth`)
- ✅ Log metrics (loss/win rate/lr/temperature)
- ✅ Export final ONNX model (`model_gobang.onnx`)

**Protocol Compliance**:
- ❌ No C++ calls allowed
- ❌ No inference logic included
- ❌ No community progress requiring manual updates
- ✅ Synthetic data only (zero human records)

**StarGo Migration**:
- Only modify [`config.yaml`](python/config.yaml): `BOARD_SIZE=19`, `INPUT_CHANNELS=17`
- Zero code changes, training logic fully reusable

### C++ Inference Engine

**Features**:
- **Pure inference architecture**: Zero coupling with Python training loop
- **Extreme performance**: O3 optimization + LTO + march=native + HPC scheduling
- **Meta-learning enhanced**: Dynamic adjustment of MCTS, ONNX threads, memory pools
- **Parallel search**: Multi-threaded MCTS, fully utilizing multi-core CPUs

**Build Modes**:
| Mode | Purpose | Compile Command |
|------|---------|-----------------|
| **Debug** | Debugging, development | `cmake .. -DCMAKE_BUILD_TYPE=Debug` |
| **Release** | Competition, inference | `cmake .. -DCMAKE_BUILD_TYPE=Release` |

### Python-C++ Interface Adapter

**File**: [`python/cpp_adapter.py`](python/cpp_adapter.py)

**Core Functions**:
- **Feature plane encoding**: Fully consistent with C++ Board::create_feature_tensor (10 channels)
- **ONNX model export optimization**: Opset 14, fixed shape, constant folding
- **C++ binary loading**: Auto-detect release/debug, prioritize release
- **Backward compatible**: Support downgrade to 3-channel basic mode

---

## Tech Stack

### Python Side
- Python 3.9+
- PyTorch 2.11.0+ (CUDA 13.0)
- NumPy 2.4.3+
- ONNX Runtime 1.24.4+
- scikit-learn 1.8.0+

### C++ Side
- C++20
- ONNX Runtime v1.16.3+ (MIT)
- CMake 3.15+
- GCC 11+ or Clang 13+
- Optimization: -O3 -march=native -LTO

---

## Model Contribution Guide

### Current Status

⚠️ **No Pre-trained Models Available Yet**

This project is currently in the **algorithm verification and training framework development** phase. We have not completed large-scale training to produce high-quality pre-trained models.

### Why No Pre-trained Models?
- This is a **research prototype** focused on training methodology validation
- Complete training requires significant computational resources (GPU/TPU clusters)
- We are currently using **Rule-Based MCTS** for data generation, not neural network inference
- The training loop is designed for **algorithm verification**, not production deployment

### We Welcome Your Contributions!

🎯 **You can contribute by:**

1. **Training Your Own Models**
   - Use the provided training loop (`train_loop.py`)
   - Modify hyperparameters in `config.yaml`
   - Train on your local machine or cloud GPUs

2. **Sharing Trained Models**
   - Upload your trained checkpoints to our community repository
   - Provide training logs and performance metrics
   - Document your training configuration

3. **Improving Training Efficiency**
   - Optimize the training pipeline
   - Implement distributed training
   - Add new training techniques

### Model Submission Requirements

To ensure quality and reproducibility, submitted models must include:

#### Required Information
```
contributed_models/{category}/{model_name}/
├── model_iter_50.pth          # Model checkpoint
├── training_config.yaml       # Exact config used
├── training_log.json          # Complete training logs
├── performance_metrics.md     # Win rates, loss curves
└── README.md                  # Training details
```

#### Performance Requirements
- **Minimum Training**: At least 50 iterations (25,000 games)
- **Win Rate**: Must beat Rule-Based MCTS (>55% win rate)
- **Loss Convergence**: Show clear downward trend in training loss
- **Documentation**: Explain any modifications to the base training loop

#### Categories for Models
- `beginner/` - First attempts, learning experiments
- `intermediate/` - Solid performance, beat baseline MCTS
- `advanced/` - State-of-the-art, tournament-level play
- `experimental/` - Novel architectures or training methods

For detailed contribution guide, see [`python/MODEL_CONTRIBUTION_GUIDE.md`](python/MODEL_CONTRIBUTION_GUIDE.md).

---

## Roadmap: Transition to Go (StarGo)

This gobang implementation is a **stepping stone** for our ultimate goal: **StarGo** - a full-scale Go AI system.

**What we're validating:**
- Training loop architecture
- Feature encoding consistency
- MCTS data generation pipeline
- Model evaluation frameworks

**What will transfer to StarGo:**
- All training methodologies
- Neural network architectures (adapted for 19x19)
- Evaluation and benchmarking tools
- Community contribution processes

---

## Protocol Compliance Assurance

- Complete MIT License text in `LICENSE` file
- Training process uses no human records, compliant with MIT spirit
- All dependencies are MIT/BSD licensed, no GPL/LGPL risks

**Dependencies verified via PyPI:**
- torch (BSD-3-Clause)
- torchvision (BSD-3-Clause)
- numpy (BSD-3-Clause)
- onnx (MIT)
- onnxruntime (MIT)
- tqdm (MIT)
- scikit-learn (BSD-3-Clause)
- pyyaml (MIT)

**No GPL/LGPL components, safe for commercial/closed-source projects.**

> **Disclaimer**: This project is provided "as is" for algorithm verification and educational purposes. While licensed under MIT, it is not optimized for production deployment. For serious Go AI development, follow our future **StarGo** project.

---

## Author

StarGobang Project Team

## License

MIT License - See LICENSE file for details
