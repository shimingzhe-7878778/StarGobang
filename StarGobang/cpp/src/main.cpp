/*
 * MIT License
 * 
 * Copyright (c) 2026 StarGobang Team
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "gobang_engine.h"
#include <iostream>
#include <string>
#include <chrono>

int main(int argc, char* argv[]) {
    try {
        // Check command line arguments
        if (argc < 2) {
            std::cerr << "Usage: " << argv[0] << " <model.onnx>" << std::endl;
            std::cerr << std::endl;
            std::cerr << "Model file locations:" << std::endl;
            std::cerr << "  Python training output: StarGobang/python/models/" << std::endl;
            std::cerr << "  C++ inference input:    cpp/models/ (symlink)" << std::endl;
            std::cerr << std::endl;
            std::cerr << "Example:" << std::endl;
            std::cerr << "  " << argv[0] << " models/model_gobang.onnx" << std::endl;
            return 1;
        }
        
        std::cout << "=== Gomoku AI - Pure Inference Engine ==="<< std::endl;
        std::cout << "=========================================" << std::endl;
        std::cout << std::endl;
        
        // Create engine and load model
        gobang::GobangEngine engine(argv[1]);
        
        // Get meta learner and HPC scheduler
        auto& meta = engine.get_meta_learner();
        auto& hpc = engine.get_hpc_scheduler();
        
        // Display hardware information
        const auto& hw = meta.get_hardware();
        std::cout << "Hardware Detection:" << std::endl;
        std::cout << "  CPU Cores: " << hw.cpu_cores << std::endl;
        std::cout << "  Available Threads: " << hw.available_threads << std::endl;
        std::cout << "  Total Memory: " << hw.total_memory_mb << " MB" << std::endl;
        std::cout << "  Available Memory: " << hw.available_memory_mb << " MB" << std::endl;
        std::cout << "  GPU Support: " << (hw.has_gpu ? "Yes" : "No") << std::endl;
        std::cout << std::endl;
        
        // Set max strength mode
        std::cout << "Setting MAX_STRENGTH mode..." << std::endl;
        engine.set_performance_mode(
            gobang::MetaLearningConfig::PerformanceMode::MAX_STRENGTH
        );
        
        // Get current parameters
        const auto& params = meta.get_current_params();
        std::cout << "Optimal Parameters:" << std::endl;
        std::cout << "  MCTS Simulations: " << params.mcts_simulations << std::endl;
        std::cout << "  Parallel Threads: " << params.mcts_parallel_threads << std::endl;
        std::cout << "  ONNX Intra Threads: " << params.onnx_intra_threads << std::endl;
        std::cout << "  CPU Affinity: " << (params.use_cpu_affinity ? "Yes" : "No") << std::endl;
        std::cout << "  Memory Pool Size: " << params.memory_pool_size_mb << " MB" << std::endl;
        std::cout << std::endl;
        
        // Create board (initial state)
        gobang::Board board;
        
        // Game loop demonstration
        std::cout << "Starting game demonstration..." << std::endl;
        int move_count = 0;
        
        while (move_count < 5) {  // First 5 moves
            auto start = std::chrono::high_resolution_clock::now();
            
            // Get best move
            auto [x, y] = engine.get_best_move(board);
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            
            std::cout << "Move " << (move_count + 1) << ": (" << x << ", " << y << ") ";
            std::cout << "Time: " << duration << " ms" << std::endl;
            
            // Make move on board
            gobang::Player current = board.current_player();
            board.make_move(x, y, current);
            
            // Online adaptation after each move
            engine.online_adapt_step();
            
            move_count++;
            
            // Dynamic adjustment based on game stage
            meta.adapt_to_game_stage(
                board.move_count(),
                0.5f,  // Simplified win rate
                0.0f   // No time limit
            );
        }
        
        std::cout << "\nGame demonstration finished!" << std::endl;
        
        // Display performance metrics
        const auto& metrics = meta.get_metrics();
        std::cout << "\nPerformance Metrics:" << std::endl;
        std::cout << "  Estimated ELO: " << metrics.elo_estimate << std::endl;
        std::cout << "  Avg Inference Time: " << metrics.inference_time_ms << " ms" << std::endl;
        
        // Display HPC performance report
        auto perf_report = hpc.get_perf_report();
        std::cout << "  Total Time: " << perf_report.total_time_ms << " ms" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
