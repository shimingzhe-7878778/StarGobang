cmake_minimum_required(VERSION 3.15)
project(GobangEngine CXX)

# C++20 标准配置
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # 严格标准 C++20，禁用编译器扩展

# 添加自定义 CMake 模块路径
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# 编译器警告选项
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ONNX Runtime 依赖查找（MIT License）
find_package(ONNXRuntime REQUIRED)

# 包含目录
include_directories(include)

# 源文件列表
set(SOURCES
    src/board.cpp
    src/forbidden_move_detector.cpp
    src/mcts.cpp
    src/gobang_engine.cpp
    src/learning_monitor.cpp
    src/meta_learner.cpp      # 元学习核心
    src/hpc_scheduler.cpp     # 高性能计算调度
)

# 添加编译优化选项 (Ubuntu 22.04.5 极致性能)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # 释放模式优化
    add_compile_options(-O3 -march=native -mtune=native)
    
    # 向量化优化
    add_compile_options(-ffast-math -funroll-loops)
    
    # 链接时优化
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# 线程支持
find_package(Threads REQUIRED)

# Linux 系统库（rt, dl, m）
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_library(RT_LIBRARY rt)
    find_library(DL_LIBRARY dl)
    find_library(M_LIBRARY m)
endif()

# 链接 ONNX Runtime
add_library(gobang_engine STATIC ${SOURCES})
target_link_libraries(gobang_engine PRIVATE 
    onnxruntime 
    Threads::Threads
    ${RT_LIBRARY}
    ${DL_LIBRARY}
    ${M_LIBRARY}
)

# 设置目标包含目录
target_include_directories(gobang_engine PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# 可执行文件示例
add_executable(gobang_demo src/main.cpp)
target_link_libraries(gobang_demo PRIVATE gobang_engine onnxruntime)

# [合规声明] 本项目仅依赖 ONNX Runtime (MIT License)，无 GPL/LGPL 组件
