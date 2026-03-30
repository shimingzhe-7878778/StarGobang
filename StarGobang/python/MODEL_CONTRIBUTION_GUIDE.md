# Model Contribution Guide

**Platform**: Linux only (tested on Ubuntu 24.04)

## Current Status

⚠️ **No Pre-trained Models Available Yet**

This project is in the **algorithm verification phase**. We have not completed large-scale training to produce production-ready models.

## Why No Pre-trained Models?

1. **Research Focus**: This is a prototype for training methodology validation
2. **Resource Requirements**: Complete training needs GPU/TPU clusters
3. **Architecture Decision**: Using Rule-Based MCTS for data generation, not neural inference
4. **Purpose**: Algorithm verification, not production deployment

## How You Can Contribute

### 1. Train Your Own Models

Use the provided training framework:

```bash
cd python
source .venv/bin/activate
python3 train_loop.py
```

Modify `config.yaml` for different training strategies.

### 2. Share Your Results

Upload your trained models with complete documentation.

### 3. Improve the Pipeline

- Optimize training efficiency
- Add new techniques
- Implement distributed training

## Model Submission Requirements

### Required Files

Your submission must include:

```
contributed_models/{category}/{model_name}/
├── model_iter_50.pth          # Final model checkpoint
├── training_config.yaml       # Exact configuration used
├── training_log.json          # Complete training logs
├── performance_metrics.md     # Win rates, loss curves
├── evaluation_results.json    # Benchmark results
└── README.md                  # Training details
```

### Performance Requirements

**Minimum Standards:**
- ✅ At least 50 training iterations (25,000 games)
- ✅ Beat Rule-Based MCTS (>55% win rate)
- ✅ Show loss convergence
- ✅ Document all modifications

**Evaluation Metrics:**
```python
# Run benchmark against Rule-Based MCTS
python3 arena.py \
  --model checkpoints/your_model.pth \
  --opponent rule_based_mcts \
  --num-games 100
```

Required metrics:
- Win rate vs MCTS
- Average game length
- Move quality analysis

### Model Categories

Submit to appropriate category based on performance:

#### `beginner/` - Learning Experiments
- First training attempts
- Testing hyperparameters
- May not beat MCTS consistently
- **Purpose**: Learning and experimentation

#### `intermediate/` - Solid Performance
- Consistently beats Rule-Based MCTS (>60%)
- Shows clear improvement over iterations
- Good for general use
- **Purpose**: Baseline strong models

#### `advanced/` - Tournament Level
- State-of-the-art performance (>70% vs MCTS)
- Novel training strategies
- Optimized architectures
- **Purpose**: Pushing boundaries

#### `experimental/` - Novel Approaches
- Alternative network architectures
- New training methods
- Unconventional techniques
- **Purpose**: Innovation and research

## Submission Process

### Step 1: Prepare Your Model

```bash
# Ensure you have:
checkpoints/
├── ckpt_iter_49.pth          # Latest checkpoint
logs/
├── final_training_log.json   # Complete logs
└── training_config.yaml      # Config file
```

### Step 2: Run Benchmarks

```bash
# Evaluate against Rule-Based MCTS
python3 eval_model.py \
  --model checkpoints/ckpt_iter_49.pth \
  --games 100 \
  --output evaluation_results.json
```

### Step 3: Create Documentation

In `README.md`, include:

```markdown
# Model Name

## Training Configuration
- Iterations: 50
- Games per iteration: 500
- MCTS simulations: 800
- Learning rate: 0.0001
- Residual blocks: 10

## Performance
- Win rate vs MCTS: 65%
- Training time: 2 hours (RTX 3060)
- Final loss: 0.234

## Modifications
[List any changes to base training loop]

## Notes
[Any additional information]
```

### Step 4: Submit Pull Request

1. Fork the repository
2. Create branch: `model-{category}-{name}`
3. Add files to `contributed_models/{category}/{name}/`
4. Update main README with your model info
5. Submit PR with detailed description

### Step 5: Community Review

- Maintainers will review your submission
- Validate performance claims
- Check documentation completeness
- Merge if standards are met

## Quality Control

To maintain high standards:

### Verification Process
1. **Reproducibility**: Others should replicate results
2. **Performance**: Must meet minimum requirements
3. **Documentation**: Clear and complete
4. **Code Quality**: Follow project standards

### Maintenance
- Model owners should respond to issues
- Provide training support
- Update if bugs found
- Help community members

## Recognition

Contributors receive:
- ✅ Credit in README.md
- ✅ GitHub contributor status
- ✅ Citation in research papers
- ✅ Community recognition

## Roadmap to StarGo

Your contributions help validate techniques for **StarGo** - our Go AI system.

**What transfers:**
- Training methodologies
- Evaluation frameworks
- Community processes
- Best practices

**What changes:**
- Board size: 15x15 → 19x19
- Input channels: 10 → 17 (AlphaGo features)
- Network scale: Larger models
- Training duration: Longer runs

## Questions?

Open an issue for:
- Training problems
- Submission questions
- Feature requests
- Bug reports

---

**Thank you for contributing to open-source AI research!** 🚀

Every model helps us understand training dynamics better and moves us closer to StarGo.
