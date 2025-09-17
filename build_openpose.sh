#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
OP_DIR="$REPO_DIR/precompute/pose/openpose"

mkdir -p "$OP_DIR/build"
cd "$OP_DIR/build"

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_PYTHON=ON \
  -DBUILD_CAFFE=ON \
  -DUSE_CUDNN=ON \
  -DBUILD_EXAMPLES=OFF \
  -DCMAKE_BUILD_RPATH='\$ORIGIN:\$ORIGIN/../../caffe/lib:\$ORIGIN/../../caffe/src/openpose_lib-build/lib'

make -j"$(nproc)"

