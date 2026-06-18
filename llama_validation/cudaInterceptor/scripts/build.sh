#!/usr/bin/env bash
# Exit immediately if any command fails (saves debugging time!)
set -e

ROOT=~/"FYP/llama_validation/cudaInterceptor"

echo "🧹 Cleaning and creating build directory..."
rm -rf "$ROOT/build"
mkdir -p "$ROOT/build"

echo "📦 Building static library libphaseguard.a..."
# Compile .cpp files from both subdirectories
g++ -fPIC -c -O2 -std=c++17 \
    -I "$ROOT/phaseGuard/include" \
    "$ROOT/phaseGuard/src/phaseLib/"*.cpp \
    "$ROOT/phaseGuard/src/policy/"*.cpp

ar rcs "$ROOT/build/libphaseguard.a" *.o
rm *.o

echo "🚀 Building shared object preload library..."
g++ -fPIC -shared -O2 -std=c++17 \
    -I "$ROOT/phaseGuard/include" \
    -I /usr/local/cuda/include \
    "$ROOT/syncInterceptor.cpp" \
    -L "$ROOT/build" -lphaseguard \
    -ldl -lpthread -lrt \
    -o "$ROOT/build/libgpuphase_preload.so"

echo "🎉 Build finished successfully!"

