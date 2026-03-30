# StarGobang AI Training & Evaluation System

## Overview
Complete Gobang (Five-in-a-Row) AI training and evaluation system featuring **self-iterative training loop** (pure training domain · zero inference code).

### Core Mission
> **Python Side = Self-evolving training loop (from zero to strong) | C++ Side = Millisecond-level inference engine**  
> **Loop Value: Not for grinding Elo, but to forge a verifiable, transferable training pipeline for StarGo**

### Important Notices

#### 🎯 **Purpose: Algorithm Verification Only**
This project is designed **solely for algorithm verification and research purposes**. It is **NOT intended for production-level deployment** or commercial use.

#### ⚖️ **License: MIT**
All code in this repository is released under the **MIT License**, ensuring maximum freedom for research, experimentation, and derivative works.

#### 🚀 **Future Direction: Transition to Go (StarGo)**
This gobang implementation serves as a **stepping stone and testing ground** for our ultimate goal: developing **StarGo** - a full-scale Go AI system. The architectures, training methodologies, and verification frameworks developed here will directly inform and validate the approaches used in StarGo, where true strategic depth will be realized.

## License Declaration
**All code in this project uses MIT License. Dependencies verified via PyPI:**
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

## Directory Structure
```
python/
├── game.py                     # Forbidden move detection (fully consistent with C++)
├── model.py                    # PyTorch model definition (outputs policy[225]+value[1])
├── train_loop.py               # Self-iterative training loop main program (latest)
├── config.yaml                 # Configuration file
├── cpp_adapter.py              # Python-C++ interface adapter layer (debug/release dual-mode support)
├── utils/
│   ├── data_generator.py       # Synthetic data generation (no human records)
│   └── monitor_parser.py       # Parse JSON learning packages exported from C++
├── test_cpp_adapter.py         # C++ adapter functional tests (legacy)
├── test_cpp_binary_loader.py   # C++ binary loader tests (legacy)
├── requirements.txt            # Dependency lock list
├── LICENSE                     # Full MIT License text
└── README.md                   # This file
```

## Quick Start

### 1. Install Dependencies

**Important**: This project currently **only supports Linux** (tested on Ubuntu 24.04).

```bash
pip install -r requirements.txt
```

### 2. Run Training Loop (Optional)

```bash
python3 train_loop.py
```

**Note**: Training takes 45-60 minutes for 50 iterations.
You can skip this and use community-contributed models instead.

### 3. Interactive CLI (Play with AI!)
```bash
python3 train_loop.py
```

Configuration parameters in [`config.yaml`](config.yaml).

**Outputs**:
- Training logs: `logs/training_log.json`
- Model checkpoints: `checkpoints/ckpt_iter_X.pth`
- Final ONNX model: `model_gobang.onnx` (directly loadable by C++ side)

### 4. Verify C++ Adapter
```bash
# Test feature encoding and model export
python3 test_cpp_adapter.py

# Test C++ binary loading (release priority, debug fallback)
python3 test_cpp_binary_loader.py
```

**Play with AI**:
```bash
# Play against trained model
python3 cli.py play --model checkpoints/ckpt_iter_49.pth

# Watch AI vs AI battle
python3 cli.py watch --model1 model1.pth --model2 model2.pth --games 3

# View model information
python3 cli.py info --model checkpoints/model.pth
```

See [`CLI_GUIDE.md`](CLI_GUIDE.md) for detailed instructions.

**C++ Loading Strategy**:
- Priority: `cmake-build-release/libgobang_engine.a` (best performance)
- Fallback: `cmake-build-debug/libgobang_engine.a` (with debug info)
- Legacy: `build/libgobang_engine.a` (old version)

## Core Features

### Self-Iterative Training Loop
**File**: [`train_loop.py`](train_loop.py) + [`config.yaml`](config.yaml)

**Three Elements of the Loop**:
1. **Self-play data generation** - MCTS self-play using current model
2. **Train new model** - Update based on generated data
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
- Only modify [`config.yaml`](config.yaml): `BOARD_SIZE=19`, `INPUT_CHANNELS=17`
- Zero code changes, training logic fully reusable

### Python-C++ Interface Adapter
**File**: [`cpp_adapter.py`](cpp_adapter.py)

**Core Functions**:
- **Feature plane encoding** - Fully consistent with C++ Board::create_feature_tensor (10 channels)
- **ONNX model export optimization** - Opset 14, fixed shape, constant folding
- **C++ binary loading** - Auto-detect release/debug, prioritize release
- **Backward compatible** - Support downgrade to 3-channel basic mode

**C++ Loading Strategy**:
```python
from cpp_adapter import get_cpp_loader

loader = get_cpp_loader()
if loader:
    print(f"Loading mode: {loader.build_mode}")  # release or debug
    flags = loader.get_compiler_flags()  # Get compiler flags
```

**Test Verification**:
- ✅ Feature encoding consistency test
- ✅ Model export test
- ✅ C++ library loading test
- ✅ Fallback mechanism test

### Data Generation Compliance
- Only random strategy + rule filtering for data generation
- No human record files loaded (.sgf/.txt)

## Protocol Compliance Assurance
- Complete MIT License text in `LICENSE` file
- Training process uses no human records, compliant with MIT spirit
- All dependencies are MIT/BSD licensed, no GPL/LGPL risks

## Tech Stack
- Python 3.9+
- PyTorch 2.11.0+ (CUDA 13.0)
- NumPy 2.4.3+
- ONNX Runtime 1.24.4+
- scikit-learn 1.8.0+

## Author
StarGobang Project Team

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
├── model_name.pth              # Model checkpoint
├── training_config.yaml        # Exact config used for training
├── training_log.json           # Complete training logs
├── performance_metrics.md      # Win rates, loss curves, etc.
└── README.md                   # Training details and notes
```

#### Performance Requirements
- **Minimum Training**: At least 50 iterations (25,000 games)
- **Win Rate**: Must beat Rule-Based MCTS (>55% win rate)
- **Loss Convergence**: Show clear downward trend in training loss
- **Documentation**: Explain any modifications to the base training loop

#### Submission Process
1. Fork the repository
2. Create a branch: `model-contribution-{your_name}`
3. Add your models to `contributed_models/{category}/`
4. Submit a Pull Request with detailed description
5. Wait for community review and validation

#### Categories for Models
- `beginner/` - First attempts, learning experiments
- `intermediate/` - Solid performance, beat baseline MCTS
- `advanced/` - State-of-the-art, tournament-level play
- `experimental/` - Novel architectures or training methods

### Roadmap: Transition to Go (StarGo)

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

## Roadmap
This gobang system is a **prototype and validation platform** for techniques that will be applied to **StarGo** - our upcoming Go AI project. The real strategic challenge begins with Go.

## License
MIT License - See LICENSE file for details
