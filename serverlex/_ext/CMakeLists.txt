# serverlex/_ext/CMakeLists.txt
# ─────────────────────────────────────────────────────────────────────────────
# CMake build for the Serverlex native C++ extension.
#
# Quick start (from repo root):
#   cmake -B build -S serverlex/_ext -DCMAKE_BUILD_TYPE=Release
#   cmake --build build
#   cp build/_serverlex_ext*.so serverlex/
# ─────────────────────────────────────────────────────────────────────────────
cmake_minimum_required(VERSION 3.18)
project(serverlex_ext LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Python & pybind11 ────────────────────────────────────────────────────────
find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# ── OpenMP (optional — enables parallel PSI across columns) ──────────────────
find_package(OpenMP)

# ── Extension module ─────────────────────────────────────────────────────────
pybind11_add_module(_serverlex_ext psi_kernel.cpp)

# Aggressive optimisation flags
target_compile_options(_serverlex_ext PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:
        -O3 -march=native -ffast-math -funroll-loops>
    $<$<CXX_COMPILER_ID:MSVC>:
        /O2 /fp:fast /GR->
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(_serverlex_ext PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "✅ OpenMP enabled — batch PSI will parallelise across columns")
else()
    message(WARNING "⚠️  OpenMP not found — batch PSI will run single-threaded")
endif()

# ── Install into the serverlex package ───────────────────────────────────────
install(TARGETS _serverlex_ext
        LIBRARY DESTINATION "${CMAKE_SOURCE_DIR}/../../")
