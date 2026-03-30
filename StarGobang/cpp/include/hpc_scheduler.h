// 高性能计算调度器 - Ubuntu 22.04.5 极致性能优化
// 核心目标：在无 sudo 权限下最大化 CPU/GPU 算力释放
// 技术手段：线程调度优化、CPU 亲和性、内存复用、并行计算

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
#ifndef GOBANG_HPC_SCHEDULER_H
#define GOBANG_HPC_SCHEDULER_H

#include "meta_learner.h"
#include <vector>
#include <thread>
#include <atomic>
#include <functional>
#include <memory>

namespace gobang {

// CPU 亲和性配置
struct CPUAffinityConfig {
    std::vector<int> core_ids;          // 绑定的 CPU 核心 ID
    bool enable_hyper_threading;        // 启用超线程
    int numa_node_id;                   // NUMA 节点 ID
    bool isolate_master_thread;         // 隔离主线程
    
    CPUAffinityConfig();
};

// 内存池配置
struct MemoryPoolConfig {
    size_t pool_size_mb;                // 内存池大小 (MB)
    size_t block_size_bytes;            // 块大小 (字节)
    bool enable_huge_pages;             // 启用大页 (如支持)
    bool preallocate;                   // 预分配
    bool enable_defrag;                 // 启用碎片整理
    
    MemoryPoolConfig();
};

// 线程池配置
struct ThreadPoolConfig {
    int num_threads;                    // 线程数量
    int stack_size_mb;                  // 栈大小 (MB)
    int priority;                       // 优先级 (-20~19, 需要权限，简化为 0-3)
    bool spin_wait;                     // 自旋等待 (低延迟)
    std::chrono::milliseconds idle_timeout; // 空闲超时
    
    ThreadPoolConfig();
};

// 任务调度器
class TaskScheduler {
public:
    explicit TaskScheduler(const ThreadPoolConfig& config);
    ~TaskScheduler();
    
    // 提交任务
    template<typename F>
    void submit(F&& task);
    
    // 并行执行任务
    template<typename F>
    void parallel_for(size_t start, size_t end, F&& func);
    
    // 等待所有任务完成
    void wait_all();
    
    // 获取活跃线程数
    int get_active_thread_count() const;
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// 内存池管理器
class MemoryPoolManager {
public:
    explicit MemoryPoolManager(const MemoryPoolConfig& config);
    ~MemoryPoolManager();
    
    // 分配内存
    void* allocate(size_t size);
    
    // 释放内存
    void deallocate(void* ptr);
    
    // 获取统计信息
    size_t get_used_memory() const;
    size_t get_pool_size() const;
    float get_utilization() const;
    
    // 优化内存布局
    void optimize_layout();
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// 高性能计算调度器 - 统一管理所有性能优化
class HPCScheduler {
public:
    static HPCScheduler& instance();  // 单例模式
    
    // 初始化 (调用一次)
    void initialize(const HardwareResources& hw);
    
    // 关闭
    void shutdown();
    
    // 获取 CPU 亲和性配置
    const CPUAffinityConfig& get_cpu_affinity() const;
    
    // 设置 CPU 亲和性 (无需 sudo 的优化方式)
    void apply_cpu_affinity(int thread_id = -1);  // -1 表示当前线程
    
    // 获取线程池
    TaskScheduler& get_task_scheduler();
    
    // 获取内存池
    MemoryPoolManager& get_memory_pool();
    
    // 获取硬件资源
    const HardwareResources& get_hardware() const;
    
    // 性能分析开始/结束
    void start_perf_profile(const std::string& name);
    void end_perf_profile(const std::string& name);
    
    // 获取性能报告
    struct PerfReport {
        double total_time_ms;
        double cpu_time_ms;
        size_t memory_allocations;
        size_t cache_misses;
        float ipc;  // Instructions per cycle
    };
    
    PerfReport get_perf_report() const;
    
    // 动态调整线程数
    void adjust_thread_count(int new_count);
    
    // 获取最优线程数
    int get_optimal_thread_count() const;
    
private:
    HPCScheduler();
    ~HPCScheduler();
    HPCScheduler(const HPCScheduler&) = delete;
    HPCScheduler& operator=(const HPCScheduler&) = delete;
    
    struct Impl;
    std::unique_ptr<Impl> impl_;
    
    void detect_hardware();
    void optimize_thread_schedule();
    void setup_memory_pool();
};

// RAII 性能分析器
class PerfProfiler {
public:
    explicit PerfProfiler(const std::string& name);
    ~PerfProfiler();
    
private:
    std::string name_;
    std::chrono::high_resolution_clock::time_point start_;
};

// 宏定义用于方便的性能分析
#define HPC_PROFILE(name) \
    gobang::PerfProfiler _profiler_##__LINE__(name)

} // namespace gobang

#endif // GOBANG_HPC_SCHEDULER_H
