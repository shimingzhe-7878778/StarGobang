# Model Template README

Copy this template and fill in your model details.

---

# [Model Name]

**Category:** [beginner | intermediate | advanced | experimental]  
**Author:** [Your Name/GitHub username]  
**Date:** [YYYY-MM-DD]  
**Training Duration:** [X hours/days]

## Quick Stats

| Metric | Value |
|--------|-------|
| **Win Rate vs MCTS** | XX% |
| **Training Iterations** | XX |
| **Total Games** | XX,XXX |
| **Final Loss** | X.XXX |
| **Avg Game Length** | XX moves |

## Training Configuration

```yaml
# Copy your exact config.yaml here
TOTAL_ITERATIONS: 50
GAMES_PER_ITER: 500
EVAL_INTERVAL: 5
MCTS_SIMULATIONS: 800
TEMPERATURE: 1.0

MODEL:
  INPUT_CHANNELS: 10
  HIDDEN_CHANNELS: 256
  NUM_RESIDUAL_BLOCKS: 10

TRAINING:
  BATCH_SIZE: 64
  EPOCHS_PER_ITER: 5
  LEARNING_RATE: 0.0001
```

## Performance Metrics

### Win Rate Progression

| Iteration | Win Rate vs MCTS | Loss |
|-----------|------------------|------|
| 10        | XX%              | X.XX |
| 20        | XX%              | X.XX |
| 30        | XX%              | X.XX |
| 40        | XX%              | X.XX |
| 50        | XX%              | X.XX |

### Loss Curves

![Loss Curve](loss_curve.png) *(optional: add screenshot)*

## Modifications to Base Training

[List any changes you made to the training loop:]

- [ ] Modified network architecture
- [ ] Changed learning rate schedule
- [ ] Added regularization
- [ ] Adjusted MCTS parameters
- [ ] Other: [describe]

**Details:**

[Explain your modifications here]

## Hardware Used

- **GPU:** [e.g., RTX 3060, A100, etc.]
- **CPU:** [e.g., Ryzen 9 5900X]
- **RAM:** [e.g., 32GB]
- **Training Time:** [e.g., 2 hours]

## Evaluation Details

```bash
# Command used for evaluation
python3 eval_model.py \
  --model checkpoints/ckpt_iter_49.pth \
  --games 100 \
  --output evaluation_results.json
```

### Sample Games

[Optional: Include interesting game positions or analysis]

## How to Use This Model

```bash
# Load this model in your code
from model import load_checkpoint, GobangNet

model = GobangNet(input_channels=10, hidden_channels=256, num_residual_blocks=10)
load_checkpoint('path/to/this/model.pth', model, None)
```

## Known Limitations

[Be honest about weaknesses:]

- Struggles with opening patterns
- Endgame could be stronger
- Vulnerable to specific strategies
- etc.

## Future Improvements

[What could make this model better:]

- More training iterations
- Larger network architecture
- Better hyperparameters
- Data augmentation
- etc.

## Acknowledgments

- Based on StarGobang training framework
- Used Rule-Based MCTS as baseline
- Thanks to [anyone who helped]

## Contact

- GitHub: [@username]
- Email: [your@email.com]
- Discord: [your_discord#1234]

---

**Model Category Guide:**

- **beginner**: Learning experiments, may not beat MCTS consistently
- **intermediate**: Beats MCTS >60%, solid performance
- **advanced**: Beats MCTS >70%, tournament level
- **experimental**: Novel approaches, alternative architectures

**Thank you for contributing!** 🚀
