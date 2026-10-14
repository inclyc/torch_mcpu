cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(TORCH_MCPU CXX C)

include(GNUInstallDirs)
include(CheckCXXCompilerFlag)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)

option(TORCH_MCPU_ENABLE_MEMORY_PROTECTION
       "Enable page-level protection for simulated mcpu device memory" OFF)
option(TORCH_MCPU_ENABLE_CPU_FALLBACK
       "Enable boxed CPU fallback for unimplemented mcpu operators" ON)
option(TORCH_MCPU_ENABLE_SYNC_KERNEL_LAUNCH
       "Synchronize the launch stream after every mcpu kernel launch" OFF)
option(TORCH_MCPU_SHARED_USE_PREMAPPED
       "Use the pre-mapped shared arena instead of the Linux mmap simulator" OFF)

if(APPLE)
  set(CMAKE_INSTALL_RPATH "@loader_path/lib;@loader_path;${PYTORCH_INSTALL_DIR}/lib")
elseif(UNIX)
  set(CMAKE_INSTALL_RPATH "$ORIGIN/lib:$ORIGIN:${PYTORCH_INSTALL_DIR}/lib")
elseif(WIN32)
  set(CMAKE_INSTALL_RPATH "")
endif()
set(CMAKE_INSTALL_LIBDIR lib)
set(CMAKE_INSTALL_MESSAGE NEVER)

set(Torch_DIR ${PYTORCH_INSTALL_DIR}/share/cmake/Torch)
find_package(Torch REQUIRED)

if(DEFINED PYTHON_INCLUDE_DIR)
  include_directories(${PYTHON_INCLUDE_DIR})
else()
  message(FATAL_ERROR "Cannot find Python directory")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include(${PROJECT_SOURCE_DIR}/cmake/TorchPythonTargets.cmake)

add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/openreg)
add_subdirectory(${PROJECT_SOURCE_DIR}/csrc)
add_subdirectory(${PROJECT_SOURCE_DIR}/torch_mcpu/csrc)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
    PATTERN "*.hpp"
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/csrc/runtime/
    DESTINATION include/runtime
    FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/third_party/openreg/include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
    PATTERN "*.inl"
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/third_party/openreg/include/
    DESTINATION include/openreg
    FILES_MATCHING PATTERN "*.h"
    PATTERN "*.inl"
)
