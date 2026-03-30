"""
Model Evaluation and Benchmarking Script

Evaluates trained models against Rule-Based MCTS and generates performance metrics.

Usage:
    python3 eval_model.py --model checkpoints/ckpt_iter_49.pth --games 100
"""

import argparse
import json
import numpy as np
from tqdm import tqdm
import torch
import yaml
from datetime import datetime

from game import Board, BLACK, WHITE, check_win, get_legal_moves
from model import load_checkpoint, board_to_tensor, GobangNet
from train_loop import RuleBasedMCTS


def evaluate_model(model_path: str, num_games: int = 100):
    """
    Evaluate trained model against Rule-Based MCTS
    
    Args:
        model_path: Path to model checkpoint
        num_games: Number of evaluation games
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model: {model_path}")
    device = torch.device('cpu')
    model = GobangNet(input_channels=10, hidden_channels=256, num_residual_blocks=10)
    checkpoint = load_checkpoint(model_path, model, None)
    model.to(device)
    model.eval()
    print(f"✓ Model loaded (iteration {checkpoint['epoch']})")
    
    # Create opponents
    model_agent = model
    mcts_agent = RuleBasedMCTS(num_simulations=800, temperature=0.5)
    
    # Evaluation results
    results = {
        'model_wins': 0,
        'mcts_wins': 0,
        'draws': 0,
        'games_played': 0,
        'first_player_wins': 0,
        'second_player_wins': 0,
        'game_lengths': [],
        'move_history': []
    }
    
    print(f"\nStarting evaluation: {num_games} games")
    print(f"  Model (Neural Network) vs Rule-Based MCTS")
    print(f"  Temperature: 0.5 (greedy for evaluation)")
    print()
    
    for game_idx in tqdm(range(num_games), desc="Evaluating"):
        board = Board()
        current_player = BLACK if game_idx % 2 == 0 else WHITE
        winner = None
        
        # Alternate first move
        model_is_first = (game_idx % 2 == 0)
        
        move_count = 0
        game_moves = []
        
        while True:
            if move_count > 225:  # Board full
                break
            
            if current_player == BLACK:
                if model_is_first:
                    # Model plays as Black
                    tensor = board_to_tensor(board, BLACK).to(device)
                    with torch.no_grad():
                        policy, _ = model(tensor)
                    move_idx = torch.argmax(policy[0]).item()
                    x = move_idx // 15
                    y = move_idx % 15
                else:
                    # MCTS plays as Black
                    move = mcts_agent.select_move(board, BLACK)
                    if move is None:
                        break
                    x, y = move
            else:
                if not model_is_first:
                    # Model plays as White
                    tensor = board_to_tensor(board, WHITE).to(device)
                    with torch.no_grad():
                        policy, _ = model(tensor)
                    move_idx = torch.argmax(policy[0]).item()
                    x = move_idx // 15
                    y = move_idx % 15
                else:
                    # MCTS plays as White
                    move = mcts_agent.select_move(board, WHITE)
                    if move is None:
                        break
                    x, y = move
            
            # Make move
            board.make_move(x, y, current_player)
            game_moves.append((x, y, current_player))
            move_count += 1
            
            # Check win
            if check_win(board, x, y, current_player):
                winner = current_player
                break
            
            # Switch player
            current_player = WHITE if current_player == BLACK else BLACK
        
        # Record results
        results['games_played'] += 1
        results['game_lengths'].append(move_count)
        
        if winner == BLACK:
            if model_is_first:
                results['model_wins'] += 1
                results['first_player_wins'] += 1
            else:
                results['mcts_wins'] += 1
                results['second_player_wins'] += 1
        elif winner == WHITE:
            if not model_is_first:
                results['model_wins'] += 1
                results['second_player_wins'] += 1
            else:
                results['mcts_wins'] += 1
                results['first_player_wins'] += 1
        else:
            results['draws'] += 1
        
        results['move_history'].append({
            'game': game_idx,
            'moves': game_moves,
            'winner': 'draw' if winner is None else ('black' if winner == BLACK else 'white'),
            'model_first': model_is_first
        })
    
    # Calculate statistics
    total_games = results['games_played']
    model_win_rate = results['model_wins'] / total_games
    mcts_win_rate = results['mcts_wins'] / total_games
    draw_rate = results['draws'] / total_games
    avg_game_length = np.mean(results['game_lengths'])
    
    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"\nGames Played: {total_games}")
    print(f"\nWin Rates:")
    print(f"  Model (Neural Network): {results['model_wins']:3d}/{total_games} ({model_win_rate*100:5.2f}%)")
    print(f"  MCTS (Rule-Based):      {results['mcts_wins']:3d}/{total_games} ({mcts_win_rate*100:5.2f}%)")
    print(f"  Draws:                  {results['draws']:3d}/{total_games} ({draw_rate*100:5.2f}%)")
    print(f"\nFirst Player Advantage:")
    print(f"  First player wins:  {results['first_player_wins']:3d}/{total_games} ({results['first_player_wins']/total_games*100:.2f}%)")
    print(f"  Second player wins: {results['second_player_wins']:3d}/{total_games} ({results['second_player_wins']/total_games*100:.2f}%)")
    print(f"\nGame Statistics:")
    print(f"  Average game length: {avg_game_length:.1f} moves")
    print(f"  Shortest game:       {min(results['game_lengths'])} moves")
    print(f"  Longest game:        {max(results['game_lengths'])} moves")
    print("=" * 60)
    
    # Performance assessment
    print("\nPerformance Assessment:")
    if model_win_rate > 0.70:
        print(f"  🏆 EXCELLENT - Model significantly outperforms MCTS")
        print(f"     Suitable for advanced category")
    elif model_win_rate > 0.60:
        print(f"  ✓ GOOD - Model beats MCTS consistently")
        print(f"     Suitable for intermediate category")
    elif model_win_rate > 0.55:
        print(f"  ✓ PASS - Model beats MCTS (minimum requirement)")
        print(f"     Suitable for submission")
    else:
        print(f"  ⚠ NEEDS IMPROVEMENT - Model doesn't beat MCTS")
        print(f"     Consider more training iterations")
    
    print("=" * 60 + "\n")
    
    return results


def save_evaluation_results(results: dict, output_path: str, model_path: str):
    """Save evaluation results to JSON file"""
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'model_checkpoint': model_path,
        'num_games': results['games_played'],
        'metrics': {
            'model_win_rate': results['model_wins'] / results['games_played'],
            'mcts_win_rate': results['mcts_wins'] / results['games_played'],
            'draw_rate': results['draws'] / results['games_played'],
            'model_wins': results['model_wins'],
            'mcts_wins': results['mcts_wins'],
            'draws': results['draws'],
        },
        'statistics': {
            'avg_game_length': float(np.mean(results['game_lengths'])),
            'min_game_length': min(results['game_lengths']),
            'max_game_length': max(results['game_lengths']),
            'first_player_win_rate': results['first_player_wins'] / results['games_played'],
            'second_player_win_rate': results['second_player_wins'] / results['games_played'],
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Evaluation results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--games', type=int, default=100, help='Number of evaluation games')
    parser.add_argument('--output', type=str, default='evaluation_results.json', help='Output JSON file')
    
    args = parser.parse_args()
    
    # Run evaluation
    results = evaluate_model(args.model, args.games)
    
    # Save results
    save_evaluation_results(results, args.output, args.model)


if __name__ == '__main__':
    main()
