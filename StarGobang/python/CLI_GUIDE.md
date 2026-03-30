# StarGobang CLI - Command Line Interface

Interactive command-line interface for playing with trained models.

**Platform**: Linux only (tested on Ubuntu 24.04)

## Quick Start

### 1. Play Against AI

```bash
# Play against a trained model
python3 cli.py play --model checkpoints/ckpt_iter_49.pth

# Choose difficulty (not yet implemented, for future use)
python3 cli.py play --model checkpoints/model.pth --difficulty easy
```

**How to play:**
- You will be asked to choose Black or White
- Enter moves as coordinates: `x y` (e.g., `7 7`)
- Special commands:
  - `quit` or `q` - Exit the game
  - `pass` - Skip your turn

### 2. Watch AI vs AI Battle

```bash
# Watch two models play against each other
python3 cli.py watch --model1 checkpoints/model_v1.pth --model2 checkpoints/model_v2.pth

# Watch multiple games
python3 cli.py watch --model1 model1.pth --model2 model2.pth --games 5
```

**Features:**
- Colorful board display
- Move-by-move commentary
- Final statistics

### 3. View Model Information

```bash
# Show details about a trained model
python3 cli.py info --model checkpoints/ckpt_iter_49.pth
```

**Information displayed:**
- Training iteration
- Loss value
- Model parameters count
- File size

## Examples

### Example 1: First Time Playing

```bash
$ python3 cli.py play --model checkpoints/my_model.pth

============================================================
Player vs AI
============================================================

Loading model...
✓ Model loaded successfully

Choose your color:
  1. Black (you go first)
  2. White (AI goes first)
Enter choice (1/2): 1

You are playing as BLACK
Commands: 'x y' to move, 'quit' to exit, 'pass' to skip

    0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 
  ┌──────────────────────────────┐
00│ ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  · │
01│ ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  · │
...
```

### Example 2: Watching AI Battle

```bash
$ python3 cli.py watch --model1 m1.pth --model2 m2.pth --games 2

============================================================
AI vs AI Battle
============================================================

Loading models...
✓ Model 1 loaded
✓ Model 2 loaded

Game 1/2
------------------------------------------------------------

Model 1 plays (7, 7)
[Board display]

Model 2 plays (8, 8)
[Board display]

...

Model 1 wins! 🎉

Press Enter to start next game...
```

### Example 3: Checking Model Stats

```bash
$ python3 cli.py info --model checkpoints/ckpt_iter_49.pth

============================================================
Model Information
============================================================

Checkpoint Details:
  File: checkpoints/ckpt_iter_49.pth
  Training Iteration: 49
  Loss: 0.2341

Model Architecture:
  Total Parameters: 11,977,738
  Trainable Parameters: 11,977,738

File Size:
  45.75 MB

============================================================
```

## Board Display

The board uses ANSI colors for better visualization:

- **●** Black pieces (bold black)
- **○** White pieces (bold white)
- **·** Empty positions (gray)
- Last move is highlighted

Coordinates:
- Top row shows column numbers (0-14)
- Left column shows row numbers (0-14)
- Center position is (7, 7)

## Tips for Playing

### As Human Player

1. **Start in the center** - Position (7, 7) is usually strong
2. **Watch for patterns** - Look for three-in-a-row, four-in-a-row
3. **Block AI threats** - Don't let AI build long chains
4. **Think ahead** - Plan 2-3 moves in advance

### Understanding AI Evaluation

When AI plays, it shows win rate evaluation:
- **>70%** - AI has strong advantage
- **50-70%** - Slight advantage
- **30-50%** - Uncertain position
- **<30%** - AI is losing

## Performance Notes

- **CPU Mode**: All inference runs on CPU (no GPU required)
- **Move Time**: ~1 second per move
- **Memory Usage**: ~200MB during gameplay

## Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'torch'"

**Solution:**
```bash
source .venv/bin/activate
pip install torch torchvision
```

### Problem: "FileNotFoundError: [Errno 2] No such file or directory"

**Solution:**
- Check that model path is correct
- Use absolute path if needed
- Verify checkpoint file exists

### Problem: Colors not displaying correctly

**Solution:**
- Ensure your terminal supports ANSI colors
- Try using a different terminal emulator
- On Windows, use Windows Terminal or WSL

## Advanced Usage

### Customizing Display

Edit `cli.py` to modify:
- Board size (currently 15x15)
- Color schemes (ANSI codes in `ColorPrinter` class)
- Output verbosity

### Adding New Commands

Extend `main()` function in `cli.py`:
```python
# Add new subcommand
custom_parser = subparsers.add_parser('custom', help='Custom command')
custom_parser.add_argument('--arg', type=str, help='Argument')

# Implement handler
def handle_custom_command(args):
    # Your implementation
    pass
```

## Integration with Training

After training completes, you can immediately play with the model:

```bash
# Train model
python3 train_loop.py

# After training finishes, play with the result
python3 cli.py play --model checkpoints/ckpt_iter_49.pth
```

## Future Enhancements

Planned features:
- [ ] Difficulty levels (easy/medium/hard)
- [ ] Move history review
- [ ] Undo functionality
- [ ] Save/load game state
- [ ] Opening book support
- [ ] Time controls
- [ ] Tournament mode

---

**Enjoy playing with StarGobang!** 🎮

For bug reports or feature requests, please open an issue on GitHub.
