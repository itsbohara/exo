#!/bin/bash
# Reapplies the bfloat16_t / vec / complex64_t Metal JIT compatibility shim
# Run this after any uv sync or venv rebuild.
# See issues/metal-bfloat16-jit-fix.md for details.
#
# NOTE: mlx is now pinned to commit 257d5692 in pyproject.toml so uv will
# no longer auto-reinstall mlx. This script should only be needed if you
# manually run 'uv sync' or delete the venv.

set -e

KERNELS=".venv/lib/python3.13/site-packages/mlx/include/mlx/backend/metal/kernels"

if [ ! -d "$KERNELS" ]; then
  echo "Error: $KERNELS not found. Run 'uv sync' first."
  exit 1
fi

# ── bf16.h: use metal::bfloat directly ───────────────────────────────────────
BF16="$KERNELS/bf16.h"
if grep -q "metal::bfloat bfloat16_t" "$BF16" 2>/dev/null; then
  echo "bf16.h: already patched, skipping."
else
  sed -i '' 's/typedef bfloat bfloat16_t;/typedef metal::bfloat bfloat16_t;/' "$BF16"
  echo "bf16.h: patched."
fi

# ── complex.h: add bfloat16_t typedef before it is used ──────────────────────
COMPLEX="$KERNELS/complex.h"
if grep -q "metal::bfloat bfloat16_t" "$COMPLEX" 2>/dev/null; then
  echo "complex.h: already patched, skipping."
else
  sed -i '' 's/using namespace metal;/using namespace metal;\ntypedef metal::bfloat bfloat16_t;/' "$COMPLEX"
  echo "complex.h: patched."
fi

# ── utils.h: replace bare bfloat16_t and vec<> with metal:: qualified names ──
UTILS="$KERNELS/utils.h"
if grep -q "metal::bfloat>::max" "$UTILS" 2>/dev/null; then
  echo "utils.h: already patched, skipping."
else
  # Fix instantiate_float_limit(bfloat16_t) -> instantiate_float_limit(metal::bfloat)
  sed -i '' 's/instantiate_float_limit(bfloat16_t);/instantiate_float_limit(metal::bfloat);/' "$UTILS"
  # Fix inline bfloat16_t log1p and return types
  sed -i '' 's/inline bfloat16_t log1p(bfloat16_t x)/inline metal::bfloat log1p(metal::bfloat x)/' "$UTILS"
  sed -i '' 's/return Limits<bfloat16_t>::max;/return Limits<metal::bfloat>::max;/' "$UTILS"
  sed -i '' 's/return bfloat16_t(x \* (metal::log(xp1) \/ (xp1 - 1.0f)));/return metal::bfloat(x * (metal::log(xp1) \/ (xp1 - 1.0f)));/' "$UTILS"
  # Fix vec<IdxT, N> -> metal::vec<IdxT, N>
  sed -i '' 's/METAL_FUNC vec</METAL_FUNC metal::vec</g' "$UTILS"
  sed -i '' 's/^  vec<IdxT, 2> loc = {/  metal::vec<IdxT, 2> loc = {/' "$UTILS"
  sed -i '' 's/^  vec<IdxT, 3> loc = {/  metal::vec<IdxT, 3> loc = {/' "$UTILS"
  echo "utils.h: patched."
fi

echo ""
echo "All patches applied. Clearing MLX kernel cache..."
rm -rf ~/.cache/mlx
echo "Done. Run: uv run exo"
