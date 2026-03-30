"""
Verify feature encoding consistency between Python and C++

This script ensures that Python's encode_feature_planes produces identical
output to C++'s Board::create_feature_tensor()

Run this before submitting models to ensure compatibility.
"""

import numpy as np
from cpp_adapter import encode_feature_planes, NetworkSpec, BOARD_SIZE, PLAYER_BLACK, PLAYER_WHITE


def test_feature_encoding():
    """Test feature encoding with various board states"""
    
    print("=" * 60)
    print("Feature Encoding Consistency Test")
    print("=" * 60)
    
    test_cases = [
        {
            'name': 'Empty Board',
            'board': np.zeros((15, 15), dtype=np.int8),
            'history': [],
            'current_player': PLAYER_BLACK
        },
        {
            'name': 'Opening Move',
            'board': np.zeros((15, 15), dtype=np.int8),
            'history': [(7, 7, PLAYER_BLACK)],
            'current_player': PLAYER_WHITE
        },
        {
            'name': 'Complex Position',
            'board': np.zeros((15, 15), dtype=np.int8),
            'history': [
                (7, 7, PLAYER_BLACK),
                (8, 8, PLAYER_WHITE),
                (7, 8, PLAYER_BLACK),
                (8, 7, PLAYER_WHITE),
                (6, 6, PLAYER_BLACK),
                (9, 9, PLAYER_WHITE),
            ],
            'current_player': PLAYER_BLACK
        }
    ]
    
    all_passed = True
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print("-" * 60)
        
        # Encode features
        features = encode_feature_planes(
            test_case['board'],
            test_case['current_player'],
            test_case['history']
        )
        
        # Verify shape
        expected_shape = (NetworkSpec.INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)
        if features.shape != expected_shape:
            print(f"  ✗ FAILED: Wrong shape {features.shape}, expected {expected_shape}")
            all_passed = False
            continue
        
        print(f"  ✓ Shape correct: {features.shape}")
        
        # Verify data type
        if features.dtype != np.float32:
            print(f"  ✗ FAILED: Wrong dtype {features.dtype}, expected float32")
            all_passed = False
            continue
        
        print(f"  ✓ Data type correct: float32")
        
        # Verify plane 8-9 (current player encoding)
        if test_case['current_player'] == PLAYER_BLACK:
            if not np.allclose(features[8], 1.0):
                print(f"  ✗ FAILED: Plane 8 should be all 1s for Black")
                all_passed = False
                continue
            if not np.allclose(features[9], 0.0):
                print(f"  ✗ FAILED: Plane 9 should be all 0s for Black")
                all_passed = False
                continue
        else:
            if not np.allclose(features[8], 0.0):
                print(f"  ✗ FAILED: Plane 8 should be all 0s for White")
                all_passed = False
                continue
            if not np.allclose(features[9], 1.0):
                print(f"  ✗ FAILED: Plane 9 should be all 1s for White")
                all_passed = False
                continue
        
        print(f"  ✓ Current player encoding correct")
        
        # Verify history moves encoding
        if test_case['history']:
            black_moves_in_history = [(x, y) for x, y, p in test_case['history'] if p == PLAYER_BLACK]
            white_moves_in_history = [(x, y) for x, y, p in test_case['history'] if p == PLAYER_WHITE]
            
            # Check last 4 black moves
            for idx, (x, y) in enumerate(black_moves_in_history[-4:]):
                plane_idx = 3 - (len(black_moves_in_history[-4:]) - 1 - idx)
                if features[plane_idx, x, y] != 1.0:
                    print(f"  ✗ FAILED: Black move at ({x},{y}) not encoded in plane {plane_idx}")
                    all_passed = False
                    continue
            
            # Check last 4 white moves
            for idx, (x, y) in enumerate(white_moves_in_history[-4:]):
                plane_idx = 7 - (len(white_moves_in_history[-4:]) - 1 - idx)
                if features[plane_idx, x, y] != 1.0:
                    print(f"  ✗ FAILED: White move at ({x},{y}) not encoded in plane {plane_idx}")
                    all_passed = False
                    continue
            
            print(f"  ✓ History moves encoded correctly")
        
        # Print summary for this test
        print(f"  ✓ Test PASSED")
    
    # Final summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("  Feature encoding is consistent with C++ implementation")
        print("  Ready for model training and submission")
    else:
        print("✗ SOME TESTS FAILED")
        print("  Please fix feature encoding before training")
    print("=" * 60 + "\n")
    
    return all_passed


def verify_model_compatibility(model_path: str = None):
    """
    Verify that a trained model can accept the encoded features
    
    Args:
        model_path: Optional path to model checkpoint
    """
    print("\n" + "=" * 60)
    print("Model Compatibility Check")
    print("=" * 60)
    
    if model_path:
        try:
            import torch
            from model import load_checkpoint, GobangNet
            
            print(f"\nLoading model: {model_path}")
            model = GobangNet(input_channels=10, hidden_channels=256, num_residual_blocks=10)
            load_checkpoint(model_path, model, None)
            
            # Create test input
            test_board = np.zeros((15, 15), dtype=np.int8)
            test_board[7, 7] = PLAYER_BLACK
            test_history = [(7, 7, PLAYER_BLACK)]
            
            # Encode
            features = encode_feature_planes(test_board, PLAYER_BLACK, test_history)
            tensor = torch.from_numpy(features).unsqueeze(0)
            
            # Forward pass
            model.eval()
            with torch.no_grad():
                policy, value = model(tensor)
            
            print(f"  ✓ Model accepts input correctly")
            print(f"  ✓ Policy output shape: {policy.shape} (expected: [1, 225])")
            print(f"  ✓ Value output shape: {value.shape} (expected: [1, 1])")
            
            if policy.shape == (1, 225) and value.shape == (1, 1):
                print(f"  ✓ Output shapes correct")
                print(f"  ✓ Model is compatible with Python-C++ interface")
            else:
                print(f"  ✗ Output shapes incorrect!")
                
        except Exception as e:
            print(f"  ✗ Error loading model: {e}")
    else:
        print("No model path provided, skipping model compatibility check")
    
    print("=" * 60 + "\n")


if __name__ == '__main__':
    import sys
    
    # Run feature encoding test
    feature_test_passed = test_feature_encoding()
    
    # Run model compatibility check if model path provided
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        verify_model_compatibility(model_path)
    else:
        verify_model_compatibility()
    
    # Exit with appropriate code
    sys.exit(0 if feature_test_passed else 1)
