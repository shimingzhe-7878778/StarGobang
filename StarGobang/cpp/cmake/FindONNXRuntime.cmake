# ONNX Runtime CMake 配置文件
include(FindPackageHandleStandardArgs)

# 查找头文件路径
find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/include
          /usr/local/include
          /usr/include
)

# 查找库文件路径
find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/lib
         /usr/local/lib
         /usr/lib
)

# 处理包查找结果
find_package_handle_standard_args(ONNXRuntime
    REQUIRED_VARS ONNXRUNTIME_LIBRARY ONNXRUNTIME_INCLUDE_DIR
    VERSION_VAR ONNXRUNTIME_VERSION
)

# 设置目标
if(ONNXRUNTIME_FOUND)
    if(NOT TARGET onnxruntime)
        add_library(onnxruntime SHARED IMPORTED)
        set_target_properties(onnxruntime PROPERTIES
            IMPORTED_LOCATION ${ONNXRUNTIME_LIBRARY}
            INTERFACE_INCLUDE_DIRECTORIES ${ONNXRUNTIME_INCLUDE_DIR}
        )
    endif()
    
    # 设置包含目录
    set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_INCLUDE_DIR})
    set(ONNXRUNTIME_LIBRARIES ${ONNXRUNTIME_LIBRARY})
endif()

# 标记为高级选项
mark_as_advanced(ONNXRUNTIME_LIBRARY ONNXRUNTIME_INCLUDE_DIR)
