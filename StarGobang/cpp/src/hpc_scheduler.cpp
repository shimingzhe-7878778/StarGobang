// 高性能计算调度器实现 - Ubuntu 22.04.5 极致性能优化
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
#include "hpc_scheduler.h"
#include <cstring>
#include <iostream>
#include <chrono>
#include <queue>
#include <condition_variable>
#include <map>

#ifdef __linux__
#include <sched.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace gobang {

// ========== CPUAffinityConfig 实现 ==========

CPUAffinityConfig::CPUAffinityConfig()
    : enable_hyper_threading(true), numa_node_id(0), isolate_master_thread(false) {}

// ========== MemoryPoolConfig 实现 ==========

MemoryPoolConfig::MemoryPoolConfig()
    : pool_size_mb(512), block_size_bytes(64), enable_huge_pages(false),
      preallocate(true), enable_defrag(false) {}

// ========== ThreadPoolConfig 实现 ==========

ThreadPoolConfig::ThreadPoolConfig()
    : num_threads(4), stack_size_mb(2), priority(0), spin_wait(false),
      idle_timeout(std::chrono::milliseconds(100)) {}

// ========== TaskScheduler 实现 ==========

struct TaskScheduler::Impl {
    struct Task {
        std::function<void()> func;
        bool is_done;
    };
    
    std::vector<std::thread> workers;
    std::queue<Task> tasks;
    std::mutex queue_mutex;
    std::condition_variable cv;
    std::atomic<bool> stop{false};
    std::atomic<int> active_tasks{0};
    
    Impl(const ThreadPoolConfig& config) {
        for (int i = 0; i < config.num_threads; ++i) {
            workers.emplace_back([this]() {
                while (true) {
                    Task task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        cv.wait(lock, [this]() {
                            return stop.load() || !tasks.empty();
                        });
                        
                        if (stop.load() && tasks.empty()) {
                            return;
                        }
                        
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    
                    active_tasks++;
                    task.func();
                    task.is_done = true;
                    active_tasks--;
                }
            });
        }
    }
    
    ~Impl() {
        stop.store(true);
        cv.notify_all();
        for (auto& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }
};

TaskScheduler::TaskScheduler(const ThreadPoolConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

TaskScheduler::~TaskScheduler() = default;

template<typename F>
void TaskScheduler::submit(F&& task) {
    typename Impl::Task t;
    t.func = std::forward<F>(task);
    t.is_done = false;
    
    {
        std::lock_guard<std::mutex> lock(impl_->queue_mutex);
        impl_->tasks.push(std::move(t));
    }
    impl_->cv.notify_one();
}

template<typename F>
void TaskScheduler::parallel_for(size_t start, size_t end, F&& func) {
    int num_tasks = static_cast<int>(end - start);
    if (num_tasks <= 0) return;
    
    std::atomic<int> counter{0};
    
    for (size_t i = start; i < end; ++i) {
        submit([&, i]() {
            func(i);
            counter++;
        });
    }
    
    // 等待所有任务完成
    while (counter.load() < num_tasks) {
        std::this_thread::yield();
    }
}

void TaskScheduler::wait_all() {
    while (impl_->active_tasks.load() > 0 || !impl_->tasks.empty()) {
        std::this_thread::yield();
    }
}

int TaskScheduler::get_active_thread_count() const {
    return impl_->active_tasks.load();
}

// ========== MemoryPoolManager 实现 ==========

struct MemoryPoolManager::Impl {
    struct Block {
        void* ptr;
        size_t size;
        bool in_use;
    };
    
    std::vector<Block> blocks;
    std::mutex pool_mutex;
    size_t total_size;
    size_t used_size;
    
    Impl(const MemoryPoolConfig& config) : total_size(config.pool_size_mb * 1024 * 1024) {
        #ifdef __linux__
        // 分配内存池
        void* pool = mmap(nullptr, total_size, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        
        if (pool != MAP_FAILED) {
            // 预分配块
            size_t num_blocks = total_size / config.block_size_bytes;
            blocks.reserve(num_blocks);
            
            char* current = static_cast<char*>(pool);
            for (size_t i = 0; i < num_blocks; ++i) {
                blocks.push_back({
                    current,
                    config.block_size_bytes,
                    false
                });
                current += config.block_size_bytes;
            }
        }
        #endif
        
        used_size = 0;
    }
    
    ~Impl() {
        #ifdef __linux__
        if (!blocks.empty()) {
            munmap(blocks[0].ptr, total_size);
        }
        #endif
    }
    
    void* allocate(size_t size) {
        std::lock_guard<std::mutex> lock(pool_mutex);
        
        // 查找合适的块
        for (auto& block : blocks) {
            if (!block.in_use && block.size >= size) {
                block.in_use = true;
                used_size += block.size;
                return block.ptr;
            }
        }
        
        // 内存池不足，回退到系统分配
        return malloc(size);
    }
    
    void deallocate(void* ptr) {
        std::lock_guard<std::mutex> lock(pool_mutex);
        
        // 查找是否属于内存池
        for (auto& block : blocks) {
            if (block.ptr == ptr && block.in_use) {
                block.in_use = false;
                used_size -= block.size;
                return;
            }
        }
        
        // 系统分配的内存
        free(ptr);
    }
    
    void optimize_layout() {
        // 整理碎片（简化实现）
        std::lock_guard<std::mutex> lock(pool_mutex);
        // 实际实现应该移动内存块以合并空闲空间
    }
};

MemoryPoolManager::MemoryPoolManager(const MemoryPoolConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

MemoryPoolManager::~MemoryPoolManager() = default;

void* MemoryPoolManager::allocate(size_t size) {
    return impl_->allocate(size);
}

void MemoryPoolManager::deallocate(void* ptr) {
    impl_->deallocate(ptr);
}

size_t MemoryPoolManager::get_used_memory() const {
    return impl_->used_size;
}

size_t MemoryPoolManager::get_pool_size() const {
    return impl_->total_size;
}

float MemoryPoolManager::get_utilization() const {
    return static_cast<float>(impl_->used_size) / impl_->total_size * 100.0f;
}

void MemoryPoolManager::optimize_layout() {
    impl_->optimize_layout();
}

// ========== HPCScheduler 实现 ==========

struct HPCScheduler::Impl {
    HardwareResources hw;
    CPUAffinityConfig cpu_affinity;
    std::unique_ptr<TaskScheduler> task_scheduler;
    std::unique_ptr<MemoryPoolManager> memory_pool;
    
    // 性能分析
    struct PerfData {
        std::string name;
        std::chrono::high_resolution_clock::time_point start;
        double duration_ms;
    };
    
    std::vector<PerfData> perf_data;
    std::map<std::string, std::chrono::high_resolution_clock::time_point> active_profiles;
    
    PerfReport report;
};

HPCScheduler::HPCScheduler() : impl_(std::make_unique<Impl>()) {}

HPCScheduler::~HPCScheduler() = default;

HPCScheduler& HPCScheduler::instance() {
    static HPCScheduler instance;
    return instance;
}

void HPCScheduler::initialize(const HardwareResources& hw) {
    impl_->hw = hw;
    
    // 检测硬件
    detect_hardware();
    
    // 优化线程调度
    optimize_thread_schedule();
    
    // 设置内存池
    setup_memory_pool();
}

void HPCScheduler::shutdown() {
    impl_->task_scheduler.reset();
    impl_->memory_pool.reset();
}

const CPUAffinityConfig& HPCScheduler::get_cpu_affinity() const {
    return impl_->cpu_affinity;
}

void HPCScheduler::apply_cpu_affinity(int thread_id) {
    #ifdef __linux__
    cpu_set_t mask;
    CPU_ZERO(&mask);
    
    // 获取可用的 CPU 核心
    const auto& cores = impl_->cpu_affinity.core_ids;
    if (cores.empty()) {
        // 默认绑定到前 N 个核心
        int num_cores = std::min(impl_->hw.cpu_cores, impl_->hw.available_threads);
        for (int i = 0; i < num_cores; ++i) {
            CPU_SET(i, &mask);
        }
    } else {
        for (int core : cores) {
            CPU_SET(core, &mask);
        }
    }
    
    if (thread_id == -1) {
        // 当前线程
        sched_setaffinity(0, sizeof(mask), &mask);
    }
    #endif
}

TaskScheduler& HPCScheduler::get_task_scheduler() {
    if (!impl_->task_scheduler) {
        ThreadPoolConfig config;
        config.num_threads = impl_->hw.available_threads;
        impl_->task_scheduler = std::make_unique<TaskScheduler>(config);
    }
    return *impl_->task_scheduler;
}

MemoryPoolManager& HPCScheduler::get_memory_pool() {
    if (!impl_->memory_pool) {
        MemoryPoolConfig config;
        config.pool_size_mb = 512;
        impl_->memory_pool = std::make_unique<MemoryPoolManager>(config);
    }
    return *impl_->memory_pool;
}

const HardwareResources& HPCScheduler::get_hardware() const {
    return impl_->hw;
}

void HPCScheduler::start_perf_profile(const std::string& name) {
    impl_->active_profiles[name] = std::chrono::high_resolution_clock::now();
}

void HPCScheduler::end_perf_profile(const std::string& name) {
    auto it = impl_->active_profiles.find(name);
    if (it != impl_->active_profiles.end()) {
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double, std::milli>(end - it->second).count();
        
        impl_->perf_data.push_back({name, it->second, duration});
        impl_->active_profiles.erase(it);
        
        impl_->report.total_time_ms += duration;
    }
}

HPCScheduler::PerfReport HPCScheduler::get_perf_report() const {
    return impl_->report;
}

void HPCScheduler::adjust_thread_count(int new_count) {
    if (impl_->task_scheduler) {
        // 重新创建线程池
        ThreadPoolConfig config;
        config.num_threads = new_count;
        impl_->task_scheduler = std::make_unique<TaskScheduler>(config);
    }
}

int HPCScheduler::get_optimal_thread_count() const {
    // 根据负载类型返回最优线程数
    // CPU 密集型：使用所有物理核心
    // 内存密集型：使用较少的线程以避免带宽饱和
    return impl_->hw.cpu_cores;
}

void HPCScheduler::detect_hardware() {
    // 硬件检测已在 MetaLearner 中完成，这里复用
}

void HPCScheduler::optimize_thread_schedule() {
    // 应用 CPU 亲和性
    apply_cpu_affinity();
    
    // 设置线程优先级（无需 sudo 的方式）
    #ifdef __linux__
    // 尝试设置 nice 值（普通用户可设置较低优先级）
    setpriority(PRIO_PROCESS, 0, -5);  // 提高优先级（需要权限则忽略）
    #endif
}

void HPCScheduler::setup_memory_pool() {
    MemoryPoolConfig config;
    config.pool_size_mb = std::min(512, static_cast<int>(impl_->hw.available_memory_mb / 4));
    config.preallocate = true;
    
    impl_->memory_pool = std::make_unique<MemoryPoolManager>(config);
}

// ========== PerfProfiler 实现 ==========

PerfProfiler::PerfProfiler(const std::string& name)
    : name_(name), start_(std::chrono::high_resolution_clock::now()) {
    HPCScheduler::instance().start_perf_profile(name);
}

PerfProfiler::~PerfProfiler() {
    HPCScheduler::instance().end_perf_profile(name_);
}

} // namespace gobang
