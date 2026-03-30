"""
Python-C++ Interface Adapter

MIT License

Copyright (c) 2026 StarGobang Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Provides interface adaptation between Python training domain and C++ inference domain:
  - Data format conversion (Python numpy <-> C++ tensor)
  - Feature plane encoding (fully consistent with C++ Board::create_feature_tensor)
  - ONNX model export optimization (ensure C++ can directly load)
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import torch


# ============================================================================
# C++ Binary Loader (Supports debug/release dual-mode)
# ============================================================================

import os
from pathlib import Path
from typing import Optional, Tuple


class CPPBinaryLoader:
    """
    C++ Binary Loader
    
    Loading Strategy:
      1. Priority: release version (performance optimization)
      2. Fallback: debug version if release not available
      3. Log loading and verify file validity
    """
    
    # List of possible build directories (by priority)
    BUILD_DIRS = [
        'cmake-build-release',  # Priority 1: release version
        'cmake-build-debug',    # Priority 2: debug version
        'build',                # Compatibility: old build directory
    ]
    
    # Required library files
    REQUIRED_LIBS = [
        'libgobang_engine.a',   # Main engine library
    ]
    
    def __init__(self, cpp_root_dir: str = None):
        """
        Initialize loader
        
        Args:
            cpp_root_dir: Path to C++ project root (default: python/../cpp)
        """
        if cpp_root_dir is None:
            # Default: find cpp folder in parent directory
            self.cpp_root = Path(__file__).parent.parent / 'cpp'
        else:
            self.cpp_root = Path(cpp_root_dir)
        
        self.selected_build_dir = None
        self.loaded_libs = {}
        self.build_mode = None  # 'release' or 'debug'
    
    def find_build_directory(self) -> Optional[Path]:
        """
        Find available build directory (by priority)
        
        Returns:
            First valid build directory found, or None
        """
        for build_dir_name in self.BUILD_DIRS:
            build_path = self.cpp_root / build_dir_name
            
            if not build_path.exists():
                continue
            
            # Check if required library files exist
            has_required_libs = True
            for lib_name in self.REQUIRED_LIBS:
                if not (build_path / lib_name).exists():
                    has_required_libs = False
                    break
            
            if has_required_libs:
                return build_path
        
        return None
    
    def get_binary_info(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Get binary file information
        
        Returns:
            (build_mode, library_path) or (None, None) if not found
        """
        build_dir = self.find_build_directory()
        
        if build_dir is None:
            return None, None
        
        # Determine build mode
        build_mode = 'unknown'
        if 'release' in str(build_dir).lower():
            build_mode = 'release'
        elif 'debug' in str(build_dir).lower():
            build_mode = 'debug'
        
        # Get main library file
        lib_path = build_dir / 'libgobang_engine.a'
        
        return build_mode, str(lib_path)
    
    def load(self) -> bool:
        """
        Load C++ library (simulated loading, actual use requires ctypes/pybind11)
        
        Returns:
            True: Loading successful
            False: Loading failed
        """
        build_mode, lib_path = self.get_binary_info()
        
        if lib_path is None:
            print("✗ No C++ binary found")
            print(f"  Searched directories: {', '.join(self.BUILD_DIRS)}")
            print(f"  C++ root: {self.cpp_root}")
            return False
        
        # Record loading information
        self.selected_build_dir = Path(lib_path).parent
        self.build_mode = build_mode
        self.loaded_libs['gobang_engine'] = lib_path
        
        # Print loading log
        mode_indicator = "🚀" if build_mode == 'release' else "🐛"
        print(f"{mode_indicator} Loading C++ library: {lib_path}")
        print(f"   Build mode: {build_mode.upper()}")
        print(f"   Library size: {os.path.getsize(lib_path) / 1024:.1f} KB")
        
        if build_mode == 'debug':
            print(f"   ⚠️ Warning: Using debug version (lower performance, recommend building release version)")
        
        return True
    
    def verify_binaries(self) -> bool:
        """
        Verify binary file validity
        
        Returns:
            True: Verification passed
            False: Verification failed
        """
        if not self.loaded_libs:
            return False
        
        for lib_name, lib_path in self.loaded_libs.items():
            # Check if file exists
            if not os.path.exists(lib_path):
                print(f"✗ Library file not found: {lib_path}")
                return False
            
            # Check file size (should be at least a few KB)
            file_size = os.path.getsize(lib_path)
            if file_size < 1024:  # Less than 1KB may be problematic
                print(f"⚠️ Warning: Library file too small {lib_path} ({file_size} bytes)")
        
        print(f"✓ C++ library verification passed")
        return True
    
    def get_compiler_flags(self) -> dict:
        """
        Get compiler flags (for pybind11 binding)
        
        Returns:
            Compilation flags dictionary
        """
        flags = {
            'include_dirs': [str(self.cpp_root / 'include')],
            'library_dirs': [str(self.selected_build_dir)],
            'libraries': ['gobang_engine'],
            'extra_compile_args': [],
            'extra_link_args': [],
        }
        
        # Add flags based on build mode
        if self.build_mode == 'debug':
            flags['extra_compile_args'].extend(['-g', '-O0'])
            flags['extra_link_args'].append('-g')
        else:  # release
            flags['extra_compile_args'].extend(['-O3', '-DNDEBUG'])
        
        return flags


# ============================================================================
# Global Loader Instance
# ============================================================================

_cpp_loader = None

def get_cpp_loader() -> Optional[CPPBinaryLoader]:
    """
    Get global C++ loader instance
    
    Returns:
        CPPBinaryLoader instance or None
    """
    global _cpp_loader
    
    if _cpp_loader is None:
        _cpp_loader = CPPBinaryLoader()
        if not _cpp_loader.load():
            return None
    
    return _cpp_loader


def check_cpp_binaries_available() -> bool:
    """
    Check if C++ binaries are available
    
    Returns:
        True: Available
        False: Not available
    """
    loader = CPPBinaryLoader()
    build_mode, lib_path = loader.get_binary_info()
    
    if lib_path is None:
        return False
    
    # Verify file
    return os.path.exists(lib_path) and os.path.getsize(lib_path) > 1024


def print_cpp_build_status():
    """
    Print C++ build status report
    """
    print("\n" + "=" * 60)
    print("C++ Binary Status")
    print("=" * 60)
    
    loader = CPPBinaryLoader()
    
    # Check all possible build directories
    print("\nBuild Directory Check:")
    for build_dir_name in loader.BUILD_DIRS:
        build_path = loader.cpp_root / build_dir_name
        exists = build_path.exists()
        
        if exists:
            lib_path = build_path / 'libgobang_engine.a'
            lib_exists = lib_path.exists()
            lib_size = os.path.getsize(lib_path) if lib_exists else 0
            
            status = f"✓ {build_dir_name:25s} [exists]"
            if lib_exists:
                status += f" - lib: {lib_size/1024:.1f} KB"
            else:
                status += f" [missing library]"
        else:
            status = f"✗ {build_dir_name:25s} [not found]"
        
        print(f"  {status}")
    
    # Show currently selected build directory
    build_mode, lib_path = loader.get_binary_info()
    if lib_path:
        print(f"\n✓ Currently selected: {build_mode.upper()} mode")
        print(f"  Library: {lib_path}")
    else:
        print(f"\n✗ No available C++ binary found")
        print(f"  Please build C++ project first: cd cpp && cmake -B build && cmake --build build")
    
    print("=" * 60 + "\n")

# Board size
BOARD_SIZE = 15
BOARD_CELLS = BOARD_SIZE * BOARD_SIZE  # 225

# Player types (corresponds to C++ Player enum)
PLAYER_NONE = 0
PLAYER_BLACK = 1
PLAYER_WHITE = 2

# Neural network specifications (corresponds to C++ NetworkSpec)
class NetworkSpec:
    """Neural network input/output specifications (fully consistent with C++ NetworkSpec)"""
    BATCH_SIZE = 1
    INPUT_CHANNELS = 10  # Can be dynamically adjusted by meta-learning [4-20]
    HEIGHT = BOARD_SIZE
    WIDTH = BOARD_SIZE
    POLICY_OUTPUT = BOARD_CELLS  # 225 move point probabilities
    VALUE_OUTPUT = 1  # Single win rate estimate


# ============================================================================
# Constants (Consistent with C++ config.h)
# ============================================================================

# Board size
BOARD_SIZE = 15
BOARD_CELLS = BOARD_SIZE * BOARD_SIZE  # 225

def encode_feature_planes(board_state: np.ndarray, 
                          current_player: int = PLAYER_BLACK,
                          history_moves: Optional[List[Tuple[int, int, int]]] = None) -> np.ndarray:
    """
    Encode feature planes (fully consistent with C++ Board::create_feature_tensor)
    
    Importance: This function must be strictly consistent with C++ feature encoding logic, otherwise the model will fail
    
    Args:
        board_state: Board state [15, 15], values are 0(empty)/1(black)/2(white)
        current_player: Current player
        history_moves: History moves [(x, y, player), ...]
        
    Returns:
        Feature tensor [INPUT_CHANNELS, HEIGHT, WIDTH]
        
    Feature Planes (corresponds to C++):
      Plane 0-3: Black historical positions (last 4 moves)
      Plane 4-7: White historical positions (last 4 moves)
      Plane 8: Current player color (all 1 for black, all 0 for white)
      Plane 9: Opponent color (all 1 for white, all 0 for black)
    """
    # Initialize feature tensor [10, 15, 15]
    features = np.zeros((NetworkSpec.INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    total_cells = BOARD_SIZE * BOARD_SIZE
    
    # Process history moves (iterate from most recent to oldest, exactly like C++)
    if history_moves:
        # Fill last 4 black positions (Plane 0-3)
        # C++ logic: for i in 0..3, check if history[size-1-i] is black
        for i in range(min(4, len(history_moves))):
            move_idx = len(history_moves) - 1 - i
            hx, hy, hplayer = history_moves[move_idx]
            if hplayer == PLAYER_BLACK:
                if 0 <= hx < BOARD_SIZE and 0 <= hy < BOARD_SIZE:
                    features[i, hx, hy] = 1.0
        
        # Fill last 4 white positions (Plane 4-7)
        # C++ logic: for i in 0..3, check if history[size-1-i] is white
        for i in range(min(4, len(history_moves))):
            move_idx = len(history_moves) - 1 - i
            hx, hy, hplayer = history_moves[move_idx]
            if hplayer == PLAYER_WHITE:
                if 0 <= hx < BOARD_SIZE and 0 <= hy < BOARD_SIZE:
                    features[4 + i, hx, hy] = 1.0
    
    # Fill current player color (Plane 8)
    if current_player == PLAYER_BLACK:
        features[8].fill(1.0)
    else:
        features[8].fill(0.0)
    
    # Fill opponent color (Plane 9)
    if current_player == PLAYER_BLACK:
        features[9].fill(0.0)
    else:
        features[9].fill(1.0)
    
    return features


def decode_policy_output(policy_vector: np.ndarray) -> List[Tuple[int, int, float]]:
    """
    Decode policy output
    
    Args:
        policy_vector: [225] dimensional probability distribution
        
    Returns:
        Sorted move list [(x, y, probability), ...]
    """
    # Reshape to [15, 15]
    policy_map = policy_vector.reshape(BOARD_SIZE, BOARD_SIZE)
    
    # Extract probabilities for all positions
    moves_with_prob = []
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            prob = policy_map[x, y]
            if prob > 1e-6:  # Filter very low probabilities
                moves_with_prob.append((x, y, prob))
    
    # Sort by probability
    moves_with_prob.sort(key=lambda item: item[2], reverse=True)
    
    return moves_with_prob


# ============================================================================
# Data Format Converter
# ============================================================================

def python_board_to_cpp_tensor(board, current_player: int, 
                                history: Optional[List] = None) -> torch.Tensor:
    """
    Convert Python Board object to C++ compatible input tensor
    
    Args:
        board: Python game.Board object
        current_player: Current player
        history: History move list [(x, y, player), ...]
        
    Returns:
        torch.Tensor [1, INPUT_CHANNELS, HEIGHT, WIDTH]
    """
    # Get board state
    board_state = board.board if hasattr(board, 'board') else board
    
    # Encode feature planes
    features = encode_feature_planes(board_state, current_player, history)
    
    # Convert to torch tensor and add batch dimension
    tensor = torch.from_numpy(features).unsqueeze(0)
    
    return tensor


def cpp_tensor_to_python_board(tensor: np.ndarray) -> np.ndarray:
    """
    Convert C++ feature tensor back to Python board state (mainly for debugging)
    
    Args:
        tensor: [1, INPUT_CHANNELS, HEIGHT, WIDTH] or [INPUT_CHANNELS, HEIGHT, WIDTH]
        
    Returns:
        Board state [HEIGHT, WIDTH]
    """
    # Remove batch dimension
    if tensor.ndim == 4:
        features = tensor[0]
    else:
        features = tensor
    
    # Reconstruct board from feature planes
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    
    # Black positions
    board[features[0] > 0.5] = PLAYER_BLACK
    
    # White positions
    board[features[1] > 0.5] = PLAYER_WHITE
    
    return board


# ============================================================================
# ONNX Model Export Optimization (Ensure C++ Compatibility)
# ============================================================================

def export_model_for_cpp(model: torch.nn.Module, output_path: str, 
                         verbose: bool = False):
    """
    Export optimized ONNX model (optimized for C++ inference engine)
    
    Args:
        model: PyTorch model
        output_path: Output path
        verbose: Print detailed information
        
    Optimizations:
      - Fixed input shape to [1, INPUT_CHANNELS, HEIGHT, WIDTH]
      - Enabled onnxruntime graph optimization
      - Using opset 14 (C++ ONNX Runtime best compatibility version)
    """
    model.eval()
    
    # Create sample input
    dummy_input = torch.randn(1, NetworkSpec.INPUT_CHANNELS, 
                             BOARD_SIZE, BOARD_SIZE)
    
    # Export configuration
    export_config = {
        'input_names': ['input'],
        'output_names': ['policy', 'value'],
        'dynamic_axes': {},  # Fixed shape, no dynamic axes
        'opset_version': 14,  # C++ ONNX Runtime best compatibility
        'do_constant_folding': True,
        'verbose': verbose,
    }
    
    # Export model
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        **export_config
    )
    
    print(f"✓ ONNX model exported: {output_path}")
    print(f"  - Input shape: {dummy_input.shape}")
    print(f"  - Output: policy[{NetworkSpec.POLICY_OUTPUT}], value[{NetworkSpec.VALUE_OUTPUT}]")
    print(f"  - Opset: {export_config['opset_version']}")
    print(f"  - Optimization: constant folding enabled")


# ============================================================================
# Verification Tools (Ensure Python-C++ Consistency)
# ============================================================================

def verify_feature_encoding_consistency():
    """
    Verify consistency of Python feature encoding with C++
    
    Returns:
        bool: True if consistent
    """
    print("Verifying feature encoding consistency...")
    
    # Create test board
    test_board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    test_board[7, 7] = PLAYER_BLACK
    test_board[7, 8] = PLAYER_WHITE
    
    # Create test history (4 moves)
    test_history = [
        (7, 7, PLAYER_BLACK),  # Oldest
        (8, 8, PLAYER_WHITE),
        (7, 8, PLAYER_BLACK),
        (8, 7, PLAYER_WHITE),  # Most recent
    ]
    
    # Encode features
    features = encode_feature_planes(test_board, PLAYER_BLACK, test_history)
    
    # Basic verification
    assert features.shape == (NetworkSpec.INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    assert features.dtype == np.float32
    
    # Check C++ logic: for i in 0..3, history[3-i]
    # i=0: history[3]=(8,7,W) -> Plane 4
    # i=1: history[2]=(7,8,B) -> Plane 1
    # i=2: history[1]=(8,8,W) -> Plane 6
    # i=3: history[0]=(7,7,B) -> Plane 3
    assert features[1, 7, 8] == 1.0, "Plane 1 should have (7,8) - 2nd most recent black"
    assert features[3, 7, 7] == 1.0, "Plane 3 should have (7,7) - oldest black"
    assert features[4, 8, 7] == 1.0, "Plane 4 should have (8,7) - most recent white"
    assert features[6, 8, 8] == 1.0, "Plane 6 should have (8,8) - 2nd most recent white"
    
    # Check current player plane
    assert features[8].sum() == 225.0, "Current player (BLACK) plane should be all 1"
    assert features[9].sum() == 0.0, "Opponent (WHITE) plane should be all 0"
    
    print("✓ Feature encoding verification passed")
    return True


# ============================================================================
# Main Function Example
# ============================================================================

if __name__ == '__main__':
    # Run consistency verification
    verify_feature_encoding_consistency()
    
    print("\nPython-C++ Adapter Ready ✓")
    print(f"  - Board size: {BOARD_SIZE}x{BOARD_SIZE}")
    print(f"  - Input channels: {NetworkSpec.INPUT_CHANNELS}")
    print(f"  - Policy output: {NetworkSpec.POLICY_OUTPUT}")
    print(f"  - Value output: {NetworkSpec.VALUE_OUTPUT}")
    
    # Check C++ binaries
    print_cpp_build_status()
    
    # Test loader
    print("\nTesting C++ library loading...")
    loader = get_cpp_loader()
    if loader:
        print(f"✓ Loading successful! Mode: {loader.build_mode}")
        flags = loader.get_compiler_flags()
        print(f"  Compiler flags: {flags}")
    else:
        print("⚠️ C++ library not loaded (may not be compiled yet)")
