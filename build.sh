#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$DIR/build"
mkdir -p "$BUILD_DIR"

echo "=== 1. Compiling Metal Shaders ==="
xcrun -sdk macosx metal -c -O3 "$DIR/shaders/gemm_attn.metal" -o "$BUILD_DIR/gemm_attn.air"
xcrun -sdk macosx metal -c -O3 "$DIR/shaders/ops_conv.metal" -o "$BUILD_DIR/ops_conv.air"
xcrun -sdk macosx metal -c -O3 "$DIR/shaders/gemm_tiled.metal" -o "$BUILD_DIR/gemm_tiled.air"
xcrun -sdk macosx metal -c -O3 "$DIR/shaders/gemm_mpp.metal" -o "$BUILD_DIR/gemm_mpp.air"
xcrun -sdk macosx metallib "$BUILD_DIR/gemm_attn.air" "$BUILD_DIR/ops_conv.air" \
    "$BUILD_DIR/gemm_tiled.air" "$BUILD_DIR/gemm_mpp.air" -o "$BUILD_DIR/thinksound.metallib"

echo "Built $BUILD_DIR/thinksound.metallib"

echo "=== 2. Compiling C++ and Objective-C++ Sources ==="
clang++ -std=c++17 -O3 -fobjc-arc -I"$DIR/src" \
    "$DIR/src/gguf.cpp" \
    "$DIR/src/tokenizers.cpp" \
    "$DIR/src/gpu.mm" \
    "$DIR/src/net.mm" \
    "$DIR/src/clip.cpp" \
    "$DIR/src/t5.cpp" \
    "$DIR/src/dit.cpp" \
    "$DIR/src/dit_fwd.cpp" \
    "$DIR/src/vae.cpp" \
    "$DIR/src/pipeline.cpp" \
    "$DIR/tools/main.cpp" \
    -framework Metal -framework Foundation \
    -o "$BUILD_DIR/ts-native"

clang++ -std=c++17 -O3 -fobjc-arc -I"$DIR/src" \
    "$DIR/src/gguf.cpp" \
    "$DIR/src/tokenizers.cpp" \
    "$DIR/src/gpu.mm" \
    "$DIR/src/net.mm" \
    "$DIR/src/clip.cpp" \
    "$DIR/src/t5.cpp" \
    "$DIR/src/dit.cpp" \
    "$DIR/src/dit_fwd.cpp" \
    "$DIR/src/vae.cpp" \
    "$DIR/src/pipeline.cpp" \
    "$DIR/tools/parity_dump.cpp" \
    -framework Metal -framework Foundation \
    -o "$BUILD_DIR/ts-parity"

clang++ -std=c++17 -O3 -fobjc-arc -I"$DIR/src" \
    "$DIR/src/gguf.cpp" \
    "$DIR/src/tokenizers.cpp" \
    "$DIR/src/gpu.mm" \
    "$DIR/src/net.mm" \
    "$DIR/src/clip.cpp" \
    "$DIR/src/t5.cpp" \
    "$DIR/src/dit.cpp" \
    "$DIR/src/dit_fwd.cpp" \
    "$DIR/src/vae.cpp" \
    "$DIR/src/pipeline.cpp" \
    "$DIR/tools/enc_check.cpp" \
    -framework Metal -framework Foundation \
    -o "$BUILD_DIR/ts-enc-check"

clang++ -std=c++17 -O3 -fobjc-arc -I"$DIR/src" \
    "$DIR/src/gpu.mm" \
    "$DIR/tools/gemm_bench.cpp" \
    -framework Metal -framework Foundation \
    -o "$BUILD_DIR/ts-gemm-bench"

clang++ -std=c++17 -O3 -fobjc-arc -I"$DIR/src" \
    "$DIR/src/gguf.cpp" \
    "$DIR/src/gpu.mm" \
    "$DIR/src/net.mm" \
    "$DIR/tools/conv_bench.cpp" \
    -framework Metal -framework Foundation \
    -o "$BUILD_DIR/ts-conv-bench"

echo "=== Done: $BUILD_DIR/ts-native built successfully ==="
