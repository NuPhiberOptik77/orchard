#!/usr/bin/env bash
# Build the engine `orchard ai` runs: llama.cpp's own release, with the two
# patches in patches/ built into its libraries. Everything comes from public
# sources and is checked against the digests below before it is used.
#
# usage: build-engine.sh <folder>           on the Linux x86_64 machine with the model and an NVIDIA GPU
#        build-engine.sh --helper <folder>  on an Apple silicon Mac lending its GPU (no patches needed)
#
# CMAKE, JOBS and VARIANT override the cmake binary, the build jobs and the CPU
# variant (haswell suits any Intel or AMD chip with AVX2).
set -euo pipefail

TAG=b12091
COMMIT=0f35d6d7b0f548993d4323eb1df8e7ca4cf039c5
BASE=https://github.com/ggml-org/llama.cpp/releases/download/$TAG
ENGINE=llama-$TAG-bin-ubuntu-cuda-12.8-x64.tar.gz
ENGINE_SHA=e35ff86b1696b7e2b61a11839f0f33420951f2bc6e7b53840cf7b621175d0fde
CUDART=cudart-llama-$TAG-bin-ubuntu-cuda-12.8-x64.tar.gz
CUDART_SHA=d8da0f89b84b045eed31580b721206a5c6194fbf0f93272fa08f7c64054a8593
HELPER=llama-$TAG-bin-macos-arm64.tar.gz
HELPER_SHA=f1a9e3696a8d9c5950a611ca2a81f342119749d4a2cab4f6afe5d109ae1de910

CMAKE=${CMAKE:-cmake}
JOBS=${JOBS:-4}
VARIANT=${VARIANT:-haswell}
here=$(cd "$(dirname "$0")" && pwd)

helper=0
if [ "${1:-}" = --helper ]; then helper=1; shift; fi
[ $# -eq 1 ] || { sed -n 6,7p "$0" >&2; exit 2; }
folder=$1
mkdir -p "$folder"
work=$(mktemp -d "${folder%/}.build.XXXXXX")
trap 'rm -rf "$work"' EXIT

fetch() {
  curl -fL --retry 3 -o "$work/$1" "$BASE/$1"
  got=$( (sha256sum "$work/$1" 2>/dev/null || shasum -a 256 "$work/$1") | cut -d' ' -f1)
  [ "$got" = "$2" ] || { echo "$1: digest $got, expected $2" >&2; exit 1; }
  # Release archives keep their files in one top folder, or in none.
  strip=$(tar -tzf "$work/$1" | awk -F/ 'NF > 1 { n++ } END { print (n == NR) ? 1 : 0 }')
  tar -xzf "$work/$1" -C "$folder" --strip-components="$strip"
}

if [ "$helper" = 1 ]; then
  fetch "$HELPER" "$HELPER_SHA"
  echo "helper ready in $folder"
  exit 0
fi

fetch "$ENGINE" "$ENGINE_SHA"
fetch "$CUDART" "$CUDART_SHA"
git -c advice.detachedHead=false clone --quiet --depth 1 --branch "$TAG" https://github.com/ggml-org/llama.cpp "$work/src"
got=$(git -C "$work/src" rev-parse HEAD)
[ "$got" = "$COMMIT" ] || { echo "$TAG is commit $got, expected $COMMIT" >&2; exit 1; }
python3 -I "$here/patches/fast-patch.py" "$work/src"
python3 -I "$here/patches/batch-patch.py" "$work/src"
"$CMAKE" -S "$work/src" -B "$work/build" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON \
  -DGGML_CPU_ALL_VARIANTS=ON -DGGML_NATIVE=OFF -DGGML_RPC=ON -DGGML_CUDA=OFF
"$CMAKE" --build "$work/build" -j "$JOBS" --target llama ggml-base ggml-rpc "ggml-cpu-$VARIANT"
cp -P "$work"/build/bin/libllama.so* "$work"/build/bin/libggml-base.so* "$work/build/bin/libggml-rpc.so" \
  "$work/build/bin/libggml-cpu-$VARIANT.so" "$folder/"
# the loader takes whichever CPU library scores best, so leave only the patched one
find "$folder" -maxdepth 1 -name 'libggml-cpu-*.so' ! -name "libggml-cpu-$VARIANT.so" -delete
echo "engine ready in $folder"
