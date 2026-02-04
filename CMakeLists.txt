cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(qec_transformer LANGUAGES CXX)

# ============================================================================
# QEC Transformer Decoder
# Transformer-based quantum error correction for [[72,12,6]] QLDPC code
# ============================================================================

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# Find Dependencies
# ============================================================================

# Find LibTorch
# Set CMAKE_PREFIX_PATH to your LibTorch installation path:
#   cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Find HDF5 (C++ bindings required)
find_package(HDF5 REQUIRED COMPONENTS CXX)
include_directories(${HDF5_INCLUDE_DIRS})

# ============================================================================
# Source Files
# ============================================================================

set(QEC_SOURCES
    syndrome_embedding.cpp
    encoder.cpp
    decoder.cpp
    model.cpp
    data_loader.cpp
    train.cpp
    main.cpp
)

set(QEC_HEADERS
    config.h
    syndrome_embedding.h
    encoder.h
    decoder.h
    model.h
    data_loader.h
    train.h
)

# ============================================================================
# Executable
# ============================================================================

add_executable(qec_transformer ${QEC_SOURCES} ${QEC_HEADERS})

# Link libraries
target_link_libraries(qec_transformer
    "${TORCH_LIBRARIES}"
    ${HDF5_CXX_LIBRARIES}
)

# Set include directories
target_include_directories(qec_transformer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HDF5_INCLUDE_DIRS}
)

# ============================================================================
# Windows-specific: Copy DLLs
# ============================================================================

if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET qec_transformer
                       POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${TORCH_DLLS}
                       $<TARGET_FILE_DIR:qec_transformer>)
endif (MSVC)

# ============================================================================
# Build Information
# ============================================================================

message(STATUS "========================================")
message(STATUS "QEC Transformer Build Configuration")
message(STATUS "========================================")
message(STATUS "CMake version: ${CMAKE_VERSION}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "LibTorch: ${Torch_DIR}")
message(STATUS "HDF5: ${HDF5_CXX_LIBRARIES}")
message(STATUS "HDF5 include: ${HDF5_INCLUDE_DIRS}")
if(DEFINED TORCH_CUDA_VERSION)
    message(STATUS "CUDA support: ${TORCH_CUDA_VERSION}")
else()
    message(STATUS "CUDA support: Not available")
endif()
message(STATUS "========================================")
