# StarGobang AI Quick Start Guide

## 🚀 5-Minute Quick Experience

### 1. Environment Setup (1 minute)

```bash
# Navigate to project directory
cd /media/shimingzhe/StarGoBang/StarGobang/python

# Create virtual environment (if not already exists)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Installation (1 minute)

```bash
# Run module tests
python3 test_cpp_adapter.py

# Run C++ binary loader tests
python3 test_cpp_binary_loader.py
```

Expected output:
```
============================================================
Python-C++ Adapter Verification Test
============================================================

Test 1: Import cpp_adapter module
============================================================
✓ Successfully imported cpp_adapter
  - BOARD_SIZE: 15
  - INPUT_CHANNELS: 10
  - POLICY_OUTPUT: 225
  - VALUE_OUTPUT: 1

Test 2: Feature Plane Encoding
============================================================
✓ Feature shape correct: (10, 15, 15)
✓ Data type correct: float32
✓ Black position encoding correct
✓ White position encoding correct
...

Total: 5/5 tests passed
✓ All tests passed! Python-C++ adapter working properly
```

### 3. Run Training Loop (3 minutes)

```bash
# Run self-iterative training loop
python3 train_loop.py
```

After training completes, checkpoints are saved in:
```
checkpoints/
├── ckpt_iter_0.pth       # Iteration 0 checkpoint
├── ckpt_iter_4.pth       # Iteration 4 checkpoint
└── ...
logs/
├── training_log.json     # Training logs
└── final_training_log.json  # Final logs
model_gobang.onnx         # Final ONNX model (loadable by C++ side)
```

### 4. Verify C++ Adapter (Optional)

```bash
# Check C++ binary build status
python3 -c "from cpp_adapter import print_cpp_build_status; print_cpp_build_status()"
```

Output example:
```
============================================================
C++ Binary Build Status
============================================================

Build directory check:
  ✓ cmake-build-release       [exists] - lib: 961.4 KB
  ✓ cmake-build-debug         [exists] - lib: 4299.7 KB
  ✓ build                     [exists] - lib: 4299.4 KB

✓ Current selection: RELEASE mode
  Library: /path/to/cpp/cmake-build-release/libgobang_engine.a
============================================================
```

### 5. Fine-tuning Process (Optional)

```bash
# Fine-tune from JSON file exported by C++ learning_monitor
python3 fine_tune.py \
  --learning_package ../cpp/build/learning_package.json \
  --model checkpoints/ckpt_iter_0.pth \
  --output checkpoints/fine_tuned.pth
```

## 🔧 Common Commands Quick Reference

### Training Related
```bash
# Run training loop
python3 train_loop.py

# View training logs
tail -f logs/training_log.json
```

### C++ Adapter Verification
```bash
# Test feature encoding
python3 test_cpp_adapter.py

# Test binary loader
python3 test_cpp_binary_loader.py

# View build status
python3 -c "from cpp_adapter import print_cpp_build_status; print_cpp_build_status()"
```

### Model Comparison
```bash
# Multi-model ranking
python arena.py \
  --model1 checkpoints/v1.pth \
  --model2 checkpoints/v2.pth \
  --num-games 100
```

### Data Generation
```bash
# Generate data only (no training)
python -c "from utils.data_generator import generate_synthetic_data; generate_synthetic_data(1000)"
```

## 🔧 Troubleshooting

### Issue 1: No module named 'torch'
**Solution**: 
```bash
pip3 install torch torchvision
```

### Issue 2: CUDA out of memory
**Solution**: Reduce batch_size
```python
# Modify in train_loop.py
batch_size = 32  # originally 64
```

### Issue 3: C++ binary not found
**Check**:
```bash
# View build status
python3 -c "from cpp_adapter import print_cpp_build_status; print_cpp_build_status()"
```

**Solution**:
```bash
cd ../cpp
cmake -B cmake-build-release
cmake --build cmake-build-release
```

## 📊 Performance Benchmarks

### Training Speed Reference (GPU: RTX 3060)
- Training loop 50 iterations: ~30 minutes
- MCTS self-play 500 games: ~15 minutes
- ONNX export: ~5 seconds

### Model Performance
- Parameters: ~15M
- Inference speed: ~1ms/step (GPU)
- MCTS simulations: 800/sec

### C++ Binaries
- Release version: 961 KB (recommended, best performance)
- Debug version: 4.3 MB (with debug info)

## 🎯 Next Steps

1. **Adjust hyperparameters**: Edit [`config.yaml`](config.yaml)
2. **View logs**: Training process outputs detailed logs
3. **Export model**: Convert to ONNX format for C++ use (automatic)
4. **Joint debugging**: Battle testing with C++ side
5. **Verify adapters**: Run `test_cpp_adapter.py` and `test_cpp_binary_loader.py`

## 📚 More Information

- Full documentation: [README.md](README.md)
- Configuration example: [config.yaml](config.yaml)

## 💡 Tips

- For first-time runs, use smaller `--num-games` values (e.g., 100) for quick validation
- GPU acceleration significantly improves training speed
- Regularly check `checkpoints/` directory for saved models

---

**Happy training!** 🎮✨
