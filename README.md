# thinksound-native

Native Apple Silicon port of ThinkSound-1S text-to-audio, in C++17 and Metal 4. No Python, no
runtimes, no wrapper layers: GGUF weights upload straight to the GPU, every tensor op is a Metal
kernel, and the WAV comes out bit-reproducible from a seed.

Measured on an Apple M5 Max (48 GB): **8 s of audio in 8.4 s wall (RTF 1.04)** including model
load, at mel-log cosine **0.9999** against the reference implementation's golden output.

## What it is

Text-to-audio generation for video sound design. Give it a caption and a chain-of-thought
description of the sound event, it renders a mono 24 kHz WAV up to 30 s long:

- **T5 (Q8_0) + MetaCLIP (F32) text encoders** — caption goes to CLIP, the chain-of-thought
  description goes to T5, matching the reference conditioning split.
- **21-block multimodal DiT (BF16)** — joint latent/text attention streams, adaLN modulation,
  fused QKV with de-interleaved per-stream layouts.
- **Rectified-flow sampler** — Euler, 24 steps default, classifier-free guidance 5.0.
- **1D causal VAE (F32) decoder** — snake activations, real GPU residual skips, conv-transpose
  output stages.
- **GEMM and convolutions on MetalPerformancePrimitives** — `mpp::matmul2d` SIMD-group
  cooperative kernels for the dense matmuls, im2col + MPP for every conv1d, f32 accumulation
  throughout, deterministic across repeat calls (identical flows, rel-L2 0).

Every stage was ported against a C++ reference implementation and verified tap-by-tap: encoder
outputs rel-L2 1.9e-3 / 2.2e-4, DiT flow 7.7e-4, VAE 0.0054, end-to-end mel-log cosine 0.9999.

## Build

Requires macOS 26+ with Xcode 27 (MetalPerformancePrimitives headers) and nothing else. No
package manager, no dependencies.

```bash
./build.sh
```

Compiles four shaders into `build/thinksound.metallib` and the binaries:

- `build/ts-native` — the generator CLI
- `build/ts-parity` — per-tap parity dumps against reference tensors
- `build/ts-enc-check` — encoder parity checker
- `build/ts-gemm-bench`, `build/ts-conv-bench` — kernel benchmarks

## Weights

On a Fornace Mac, install the release through the private registry: one 5.8 GB download from
Cloudflare R2 with `ts-native`, `thinksound.metallib` and the six GGUF files in one folder, checked
against BLAKE3, at `~/Models/thinksound-current` (the binary's default `--dir`):

```bash
git clone https://github.com/Fornace/mac-models && mac-models/install.sh thinksound
```

By hand, put the six files from [ilintar/thinksound-gguf](https://huggingface.co/ilintar/thinksound-gguf)
(`dit-bf16`, `t5-q8`, `metaclip-text-f32`, `vae-f32`, `clip-tokenizer`, `t5-tokenizer`) and
`build/thinksound.metallib` in one directory and pass it as `--dir`. Never use `/tmp`: macOS clears it.

## Usage

```bash
build/ts-native --caption "a dog barking" \
                --cot "A dog barks several times, sharp and clear, in a quiet room." \
                --dur 8 --out dog.wav
```

| option | default | meaning |
|---|---|---|
| `--caption <text>` | `a dog barking` | short caption, CLIP conditioning |
| `--cot <text>` | (dog example) | chain-of-thought sound description, T5 conditioning; one or two sentences naming the source, the space and the texture |
| `--dir <path>` | `~/Models/thinksound-current` | directory holding the GGUF files and `thinksound.metallib` |
| `--out <path>` | `out.wav` | output WAV path |
| `--dur <sec>` | `8` | audio duration, 1..30 s |
| `--steps <int>` | `24` | Euler sampling steps |
| `--cfg <float>` | `5.0` | classifier-free guidance scale |
| `--seed <uint>` | `4242` | RNG seed; same seed and inputs give the same WAV |

Prompting guidance carried over from testing: name the **source** of the sound, the **space**
it sits in, and the **texture** (sharp / muffled / distant / close). The CoT field wants a
natural-language description of the event as a listener would recount it. Avoid stacking many
unrelated events in one generation; render one bed per scene and mix externally.

## Repository layout

```
src/         C++17 / Objective-C++ sources
  gguf.*       GGUF reader (f16/f32/q8_0)
  tokenizers.* CLIP BPE + T5 sentencepiece-style readers
  gpu.*        Metal device, library, buffer, dispatch plumbing
  net.*        GEMM/conv/norm/attention kernel launcher layer
  clip.cpp     MetaCLIP text encoder
  t5.cpp       T5 encoder (geglu, relative bias, pad mask)
  dit*.cpp     21-block MMDiT forward
  vae.cpp      1D causal VAE decoder
  pipeline.cpp end-to-end: encoders -> sampler -> VAE -> WAV
shaders/     Metal kernels (SIMD-group attention, MPP matmul, im2col conv, ops)
tools/       CLI, parity dump harness, encoder check, benchmarks
```

## Verification tools

```bash
build/ts-parity    # dumps per-tap tensors; compare against reference dumps
build/ts-enc-check # encoder rel-L2 against golden features
```

Parity gates used during the port (all green): flow 7.7e-4, joint-block text 2.3e-3,
after-fused 1.5e-3, encoders unchanged, repeated forward bit-identical, end-to-end mel-log
cosine 0.9999 vs the reference golden WAV.

## Performance notes

- 82% of wall time was the 24 CFG DiT steps before the MPP pass; now most of the remaining
  wall time is host-side weight loading and conversion, not GPU work.
- Convs run as im2col + `mpp::matmul2d` (64x32 tiles chosen by measurement: 10.6 TFLOP/s on
  the dominant DiT conv shape vs 0.40 scalar).
- The scalar GEMV keeps the small-M crown; MPP handles everything else.
- Timing methodology: warmup, interleaved runs, each run waits for an idle GPU, best of N.
  Shared-GPU contention from other jobs inflates numbers noticeably; measure under the GPU
  lock.
