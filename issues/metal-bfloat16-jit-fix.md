# Metal JIT Compile Error: `bfloat16_t` / `vec` / `complex64_t` Unknown

## Symptoms

Runner crashes at warmup with:

```
[metal::Device] Unable to build metal library from source
mlx/backend/metal/kernels/utils.h:64:25: error: unknown type name 'bfloat16_t'; did you mean 'float16_t'?
mlx/backend/metal/kernels/utils.h:144:12: error: no template named 'vec'; did you mean 'metal::vec'?
mlx/backend/metal/kernels/utils.h:73:15: error: use of undeclared identifier 'complex64_t'
```

## Root Cause

macOS 15 (Sequoia) machines that have the **macOS 26 Command Line Tools** (`26.2.0`) installed
end up with GPU compiler framework version `32023.622` in `/System/Library`.

This newer GPU compiler moved `bfloat` and `vec<>` into the `metal::` namespace exclusively
and no longer exposes them at global scope — even with `using namespace metal;`.

MLX's JIT kernel source (`bf16.h`) uses `typedef bfloat bfloat16_t` and `utils.h` uses
bare `vec<IdxT, N>`, which fail to compile against the macOS 26 GPU compiler runtime.

## Affected Setup

- macOS 15.x (Sequoia) with CLT 26.2.0 installed
- GPU compiler: `32023.622` in `/System/Library/PrivateFrameworks/GPUCompiler.framework`
- MLX version: `0.30.7.dev20260228` (exo's custom fork)
- Xcode 16.3 installed but CLT is 26.x

## Workaround Applied

Patched `.venv/lib/python3.13/site-packages/mlx/include/mlx/backend/metal/kernels/bf16.h`
to add a compatibility shim:

```diff
 using namespace metal;
 
-typedef bfloat bfloat16_t;
+// Compatibility shim for macOS 26 GPU compiler (32023.622+)
+#ifndef bfloat16_t
+typedef metal::bfloat bfloat16_t;
+#endif
+#ifndef vec
+template <typename T, int N> using vec = metal::vec<T, N>;
+#endif
```

## How to Reapply After `uv sync` / venv Rebuild

The patch lives inside the venv and will be lost if uv recreates it. Reapply with:

```bash
HEADER=.venv/lib/python3.13/site-packages/mlx/include/mlx/backend/metal/kernels/bf16.h

# Remove old typedef and insert shim
sed -i '' 's/typedef bfloat bfloat16_t;/\/\/ Compatibility shim for macOS 26 GPU compiler\n#ifndef bfloat16_t\ntypedef metal::bfloat bfloat16_t;\n#endif\n#ifndef vec\ntemplate <typename T, int N> using vec = metal::vec<T, N>;\n#endif/' "$HEADER"
```

Or just run the patch script:

```bash
bash issues/reapply-metal-patch.sh
```

## Proper Fix (macOS 26 / Tahoe only)

On **macOS 26 (Tahoe)** with Xcode 17, the official fix is either:

```bash
xcodebuild -downloadComponent metalToolchain
```

Or via **Xcode → Settings → Components → Other Components → Metal Toolchain → Get** (~704 MB).

**This option does NOT exist on macOS 15.** Xcode 16.3's Components panel shows no Metal Toolchain entry.
The `-downloadComponent` flag is also unavailable in Xcode 16.x.

On macOS 15 with CLT 26.x, the header shim above is the only workaround until either:
- MLX updates its kernel headers to use `metal::bfloat` explicitly
- Apple backports the Metal Toolchain download to Xcode 16.x

## Related

- exo issue: https://github.com/exo-explore/exo/issues/1554 (closed, macOS 26 fix only)
- MLX GPU compiler: `/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.622`
