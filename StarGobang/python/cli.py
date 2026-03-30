"""
StarGobang Interactive CLI

Simple command-line interface for playing against trained models or watching AI vs AI games.

Usage:
    python3 cli.py play --model checkpoints/model.pth     # Play against model
    python3 cli.py watch --model1 model1.pth --model2 model2.pth  # Watch AI vs AI
    python3 cli.py info --model checkpoints/model.pth     # Show model info
"""

import argparse
import sys
import numpy as np
import torch
from typing import Tuple, Optional

from game import Board, BLACK, WHITE, EMPTY, check_win, get_legal_moves
from model import load_checkpoint, GobangNet, board_to_tensor


class ColorPrinter:
    """ANSI color codes for terminal output"""
    
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    # Colors
    BLACK_PIECE = "\033[30;1m"  # Bold black
    WHITE_PIECE = "\033[37;1m"  # Bold white
    EMPTY_CELL = "\033[90m"      # Gray
    BOARD_LINE = "\033[36m"      # Cyan
    COORDINATE = "\033[33m"      # Yellow
    
    # Status colors
    SUCCESS = "\033[32m"         # Green
    WARNING = "\033[33m"         # Yellow
    ERROR = "\033[31m"           # Red
    INFO = "\033[36m"            # Cyan
    
    @classmethod
    def print(cls, text, color=None):
        if color:
            print(f"{color}{text}{cls.RESET}")
        else:
            print(text)


def print_board(board: Board, last_move: Optional[Tuple[int, int]] = None):
    """Print the board with colors and coordinates"""
    
    cp = ColorPrinter
    
    # Print column headers
    print("\n    ", end="")
    for i in range(15):
        print(f"{cp.COORDINATE}{i:2}{cp.RESET} ", end="")
    print()
    
    # Print top border
    print(f"  {cp.BOARD_LINE}┌{'─' * 30}┐{cp.RESET}")
    
    # Print rows
    for y in range(15):
        print(f"{cp.COORDINATE}{y:2}{cp.RESET} {cp.BOARD_LINE}│{cp.RESET}", end="")
        
        for x in range(15):
            cell = board.get_cell(x, y)
            
            if cell == BLACK:
                if last_move and (x, y) == last_move:
                    print(f" {cp.BLACK_PIECE}●{cp.RESET}", end="")
                else:
                    print(f" {cp.BLACK_PIECE}●{cp.RESET}", end="")
            elif cell == WHITE:
                if last_move and (x, y) == last_move:
                    print(f" {cp.WHITE_PIECE}○{cp.RESET}", end="")
                else:
                    print(f" {cp.WHITE_PIECE}○{cp.RESET}", end="")
            else:
                # Mark last move position
                if last_move and (x, y) == last_move:
                    print(f" {cp.WARNING}·{cp.RESET}", end="")
                else:
                    print(f" {cp.EMPTY_CELL}·{cp.RESET}", end="")
        
        print(f" {cp.BOARD_LINE}│{cp.RESET}")
    
    # Print bottom border
    print(f"  {cp.BOARD_LINE}└{'─' * 30}┘{cp.RESET}\n")


def get_human_move(board: Board, player: int) -> Tuple[int, int]:
    """Get move from human player via keyboard input"""
    
    cp = ColorPrinter
    
    while True:
        try:
            # Get input
            move_input = input(f"\n{'BLACK' if player == BLACK else 'WHITE'}'s turn. Enter move (x y): ").strip()
            
            # Handle special commands
            if move_input.lower() in ['quit', 'exit', 'q']:
                print(cp.WARNING + "Game aborted." + cp.RESET)
                sys.exit(0)
            
            if move_input.lower() == 'pass':
                print(cp.INFO + "Passing turn..." + cp.RESET)
                return None
            
            # Parse coordinates
            parts = move_input.split()
            if len(parts) != 2:
                print(cp.ERROR + "Invalid format. Use: x y (e.g., 7 7)" + cp.RESET)
                continue
            
            x = int(parts[0])
            y = int(parts[1])
            
            # Validate move
            if not (0 <= x < 15 and 0 <= y < 15):
                print(cp.ERROR + "Coordinates out of range (0-14)" + cp.RESET)
                continue
            
            if board.get_cell(x, y) != EMPTY:
                print(cp.ERROR + "Position already occupied" + cp.RESET)
                continue
            
            # Check for forbidden moves (for Black only)
            if player == BLACK:
                temp_board = board.copy()
                temp_board.make_move(x, y, player)
                # Note: You can add forbidden move detection here if needed
            
            return (x, y)
            
        except ValueError:
            print(cp.ERROR + "Please enter valid numbers" + cp.RESET)
        except KeyboardInterrupt:
            print(cp.WARNING + "\nGame aborted." + cp.RESET)
            sys.exit(0)


def get_ai_move(model: GobangNet, board: Board, player: int, device: torch.device) -> Tuple[int, int]:
    """Get move from AI model"""
    
    legal_moves = get_legal_moves(board, player)
    if not legal_moves:
        return None
    
    if len(legal_moves) == 1:
        return legal_moves[0]
    
    # Convert board to tensor
    tensor = board_to_tensor(board, player).to(device)
    
    # Get model prediction
    with torch.no_grad():
        policy, value = model(tensor)
    
    # Select best move
    move_idx = torch.argmax(policy[0]).item()
    x = move_idx // 15
    y = move_idx % 15
    
    # Verify move is legal
    if (x, y) not in legal_moves:
        # Fallback to first legal move
        return legal_moves[0]
    
    return (x, y)


def play_against_model(model_path: str, difficulty: str = 'normal'):
    """
    Play a game against the AI model
    
    Args:
        model_path: Path to model checkpoint
        difficulty: Difficulty level (easy/normal/hard)
    """
    
    cp = ColorPrinter
    
    print("\n" + "=" * 60)
    print(cp.BOLD + "Player vs AI" + cp.RESET)
    print("=" * 60)
    
    # Load model
    print(f"\n{cp.INFO}Loading model...{cp.RESET}")
    device = torch.device('cpu')
    model = GobangNet(input_channels=10, hidden_channels=256, num_residual_blocks=10)
    
    try:
        load_checkpoint(model_path, model, None)
        model.to(device)
        model.eval()
        print(f"{cp.SUCCESS}✓ Model loaded successfully{cp.RESET}")
    except Exception as e:
        print(f"{cp.ERROR}✗ Failed to load model: {e}{cp.RESET}")
        sys.exit(1)
    
    # Choose color
    print(f"\n{cp.INFO}Choose your color:{cp.RESET}")
    print("  1. Black (you go first)")
    print("  2. White (AI goes first)")
    
    choice = input("Enter choice (1/2): ").strip()
    human_color = BLACK if choice == '1' else WHITE
    ai_color = WHITE if human_color == BLACK else BLACK
    
    print(f"\n{cp.INFO}You are playing as {cp.BOLD}{'BLACK' if human_color == BLACK else 'WHITE'}{cp.RESET}")
    print(f"{cp.INFO}Commands: 'x y' to move, 'quit' to exit, 'pass' to skip{cp.RESET}")
    
    # Start game
    board = Board()
    current_player = BLACK
    last_move = None
    move_count = 0
    
    print_board(board)
    
    while True:
        if move_count >= 225:
            print(cp.INFO + "\nBoard is full - Draw!" + cp.RESET)
            break
        
        if current_player == human_color:
            # Human's turn
            move = get_human_move(board, current_player)
            if move is None:
                # Pass
                current_player = WHITE if current_player == BLACK else BLACK
                continue
        else:
            # AI's turn
            print(f"\n{cp.INFO}AI is thinking...{cp.RESET}")
            move = get_ai_move(model, board, current_player, device)
            
            if move:
                # Show AI's evaluation
                tensor = board_to_tensor(board, current_player).to(device)
                with torch.no_grad():
                    _, value = model(tensor)
                win_rate = (value[0, 0].item() + 1) / 2  # Normalize to [0, 1]
                print(f"{cp.INFO}AI evaluation: {win_rate*100:.1f}% win rate{cp.RESET}")
        
        if move:
            x, y = move
            board.make_move(x, y, current_player)
            last_move = (x, y)
            move_count += 1
            
            # Print board
            print_board(board, last_move)
            
            # Check win
            if check_win(board, x, y, current_player):
                winner_name = "You" if current_player == human_color else "AI"
                print(cp.BOLD + f"\n{winner_name} win! 🎉" + cp.RESET)
                break
            
            # Switch player
            current_player = WHITE if current_player == BLACK else BLACK
        else:
            # No legal moves
            print(cp.INFO + "No legal moves - Draw!" + cp.RESET)
            break
    
    print("\n" + "=" * 60)
    print(cp.INFO + "Game over!" + cp.RESET)
    print("=" * 60 + "\n")


def watch_ai_battle(model1_path: str, model2_path: str, num_games: int = 1):
    """
    Watch two AI models play against each other
    
    Args:
        model1_path: Path to first model
        model2_path: Path to second model
        num_games: Number of games to play
    """
    
    cp = ColorPrinter
    
    print("\n" + "=" * 60)
    print(cp.BOLD + "AI vs AI Battle" + cp.RESET)
    print("=" * 60)
    
    # Load models
    print(f"\n{cp.INFO}Loading models...{cp.RESET}")
    device = torch.device('cpu')
    
    model1 = GobangNet(input_channels=10, hidden_channels=256, num_residual_blocks=10)
    model2 = GobangNet(input_channels=10, hidden_channels=256, num_residual_blocks=10)
    
    try:
        load_checkpoint(model1_path, model1, None)
        print(f"{cp.SUCCESS}✓ Model 1 loaded{cp.RESET}")
    except Exception as e:
        print(f"{cp.ERROR}✗ Failed to load Model 1: {e}{cp.RESET}")
        return
    
    try:
        load_checkpoint(model2_path, model2, None)
        print(f"{cp.SUCCESS}✓ Model 2 loaded{cp.RESET}")
    except Exception as e:
        print(f"{cp.ERROR}✗ Failed to load Model 2: {e}{cp.RESET}")
        return
    
    model1.to(device)
    model2.to(device)
    model1.eval()
    model2.eval()
    
    # Statistics
    stats = {
        'model1_wins': 0,
        'model2_wins': 0,
        'draws': 0
    }
    
    # Play games
    for game_idx in range(num_games):
        print(f"\n{cp.BOLD}Game {game_idx + 1}/{num_games}{cp.RESET}")
        print("-" * 60)
        
        board = Board()
        current_player = BLACK
        last_move = None
        move_count = 0
        
        # Alternate first move
        model1_is_black = (game_idx % 2 == 0)
        
        print_board(board)
        
        while True:
            if move_count >= 225:
                print(cp.INFO + "Draw - Board full" + cp.RESET)
                stats['draws'] += 1
                break
            
            # Select model
            if (model1_is_black and current_player == BLACK) or \
               (not model1_is_black and current_player == WHITE):
                current_model = model1
                model_name = "Model 1"
            else:
                current_model = model2
                model_name = "Model 2"
            
            # Get AI move
            move = get_ai_move(current_model, board, current_player, device)
            
            if move:
                x, y = move
                board.make_move(x, y, current_player)
                last_move = (x, y)
                move_count += 1
                
                # Show move
                print(f"{model_name} plays ({x}, {y})")
                print_board(board, last_move)
                
                # Check win
                if check_win(board, x, y, current_player):
                    print(cp.BOLD + f"\n{model_name} wins! 🎉" + cp.RESET)
                    
                    if current_player == BLACK:
                        if model1_is_black:
                            stats['model1_wins'] += 1
                        else:
                            stats['model2_wins'] += 1
                    else:
                        if model1_is_black:
                            stats['model2_wins'] += 1
                        else:
                            stats['model1_wins'] += 1
                    
                    break
                
                # Switch player
                current_player = WHITE if current_player == BLACK else BLACK
            else:
                print(cp.INFO + "Draw - No legal moves" + cp.RESET)
                stats['draws'] += 1
                break
        
        # Pause between games
        if game_idx < num_games - 1:
            input(cp.INFO + "\nPress Enter to start next game..." + cp.RESET)
    
    # Final statistics
    print("\n" + "=" * 60)
    print(cp.BOLD + "Final Results" + cp.RESET)
    print("=" * 60)
    print(f"Model 1 wins: {stats['model1_wins']}")
    print(f"Model 2 wins: {stats['model2_wins']}")
    print(f"Draws:        {stats['draws']}")
    print(f"Total games:  {num_games}")
    print("=" * 60 + "\n")


def show_model_info(model_path: str):
    """Show information about a trained model"""
    
    cp = ColorPrinter
    
    print("\n" + "=" * 60)
    print(cp.BOLD + "Model Information" + cp.RESET)
    print("=" * 60)
    
    try:
        # Try to load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        print(f"\n{cp.INFO}Checkpoint Details:{cp.RESET}")
        print(f"  File: {model_path}")
        
        if 'epoch' in checkpoint:
            print(f"  Training Iteration: {checkpoint['epoch']}")
        
        if 'loss' in checkpoint:
            print(f"  Loss: {checkpoint['loss']:.4f}")
        
        # Count parameters
        if 'state_dict' in checkpoint or 'model_state_dict' in checkpoint:
            state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict'))
            total_params = sum(p.numel() for p in state_dict.values())
            trainable_params = sum(p.numel() for p in state_dict.values() if p.requires_grad)
            
            print(f"\n{cp.INFO}Model Architecture:{cp.RESET}")
            print(f"  Total Parameters: {total_params:,}")
            print(f"  Trainable Parameters: {trainable_params:,}")
        
        print(f"\n{cp.INFO}File Size:{cp.RESET}")
        import os
        file_size = os.path.getsize(model_path)
        print(f"  {file_size / 1024 / 1024:.2f} MB")
        
        print("\n" + "=" * 60 + "\n")
        
    except Exception as e:
        print(f"{cp.ERROR}Error loading model: {e}{cp.RESET}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="StarGobang Interactive CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s play --model checkpoints/model.pth
  %(prog)s watch --model1 model1.pth --model2 model2.pth --games 3
  %(prog)s info --model checkpoints/model.pth
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Play command
    play_parser = subparsers.add_parser('play', help='Play against AI model')
    play_parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    play_parser.add_argument('--difficulty', type=str, default='normal', 
                            choices=['easy', 'normal', 'hard'], help='Difficulty level')
    
    # Watch command
    watch_parser = subparsers.add_parser('watch', help='Watch AI vs AI battle')
    watch_parser.add_argument('--model1', type=str, required=True, help='First model path')
    watch_parser.add_argument('--model2', type=str, required=True, help='Second model path')
    watch_parser.add_argument('--games', type=int, default=1, help='Number of games')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show model information')
    info_parser.add_argument('--model', type=str, required=True, help='Model path')
    
    args = parser.parse_args()
    
    if args.command == 'play':
        play_against_model(args.model, args.difficulty)
    elif args.command == 'watch':
        watch_ai_battle(args.model1, args.model2, args.games)
    elif args.command == 'info':
        show_model_info(args.model)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
