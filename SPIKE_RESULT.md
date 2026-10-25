# iOS spike — PASSED (2026-10-25)

Question this spike answers: can llama.cpp be cross-compiled for the iOS **simulator**,
linked into a binary, and actually generate text there? For poidh base #1405 ("iOS version
of Kenny's Best Offline AI Research app") this was the make-or-break unknown, because on iOS
there is no `fork`/`exec` — the model has to run in-process.

## Result

```
loaded model: 494032768 params, n_vocab=151936
prompt tokens: 14
Hello, how can I assist you today?
-- done --
real  1m16.598s
```

- model: `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` (398 MB)
- toolchain: llama.cpp `v0.7.0`, CMake `CMAKE_SYSTEM_NAME=iOS`, sysroot `iphonesimulator`,
  `arm64`, deployment target 16.0, `BUILD_SHARED_LIBS=OFF`, Metal/OpenMP/BLAS off
- link: `xcrun --sdk iphonesimulator clang -target arm64-apple-ios16.0-simulator`
  against the produced `*.a` + `-lc++ -framework Accelerate`
- run: `xcrun simctl spawn <device> ./llm_probe model.gguf "<prompt>"` on macos-14 arm64 runner
- throughput in the simulator: 40 tokens in ≈31 s → **~1.3 tok/s** (CPU only, 4 threads);
  model load ≈5 s
- workflow run: `40151268110` (success), repo `9xj89gzrtw-hue/pocket-scholar-ios`

## Two bugs that cost one run each (keep them fixed)

1. `xcrun simctl spawn` does **not** inherit the workspace as cwd — every path passed to the
   spawned binary must be absolute (`$PWD/llm_probe`, `$PWD/model.gguf`).
2. `llama_tokenize(vocab, text, len, NULL, 0, ...)` returns the **negative** token count.
   Not negating it makes `malloc` fail and `llama_decode` return `n_tokens == 0`.
   Also: with the current API there is no internal `n_past`, so every decode batch needs
   explicit `pos[i]` values (prompt `0..n-1`, then one position per generated token).

## What this unlocks

A real SwiftUI app is now ordinary work: bundle `llm_probe`-style C shim as a static library,
put a `corpus` (BM25 over bundled articles, same as the Android entry) next to it, and drive
it from Swift. No network, no Play/App services, everything on device.
