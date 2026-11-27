# Inkling 2 Mini on R9700/HIP — issue #8

**Qualified on 2026-11-27 for the exact single-request, text-only configuration below.** Selected-device numerical checks and the real-model headless API proof passed. This is a bounded model-support result, not a new release, blanket AMD support or automatic qualification of fresh builds. Existing Qwen installations and release records are unchanged.

Contents: [Boundary](#qualified-boundary) · [Evidence](#results-and-evidence) · [Inputs](#immutable-inputs-and-attribution) · [Reconstruction](#reconstruction) · [Checks](#checks-and-reruns) · [Limitations](#limitations).

## Qualified boundary

| Item | Exact selection |
| --- | --- |
| Model | Inkling 2 Mini, 26B, retained-layout UD-Q4_K_XL GGUF identified below; not Inkling Small |
| Platform | Linux x86-64, kernel `7.3.4-200.fc44.x86_64` |
| GPU | One isolated Radeon AI PRO R9700, `gfx1201`, PCI `0000:03:00.0`, ROCr UUID `GPU-26508202313fa2ac` |
| Serving | Same-tree `unbiased-strata` headless HTTP adapter and `strata-ik2`, one request at a time, loopback only |
| Settings | Context 8192, prompt chunk 256, FP16 KV, all experts resident, direct-row expert decode, greedy temperature 0, reasoning effort `none`, no drafting or vision |
| Engine identity | SHA-256 `68fadf0e99e83c09e4e004a1311760cbb951a87df3917aef6af500c03a0d9f10` |
| Adapter identity | SHA-256 `63e3506b5e10c3f226b0bdbde5703b0f936b93257619ac7ed21387b2c98b855a` |
| Runtime | Existing repaired ROCr, SHA-256 `63fa08fa9461d6f5e652fa50d232116b317091dbf0bbe2204799dfd705b564ca`, bound read-only inside the qualification namespace; matching vendor libraries and kernel data in the runtime inventory |

HIP admission rejects other architectures, extra visible GPUs, Small/other geometries, context above 8192, chunks above 256, drafting, pictures, managed/CPU-offload modes and loss of the resident path. It does not expose another GPU or silently fall back to host offload. The [host kernel policy](KERNEL_POLICY.md) targets the running platform; release history is not a kernel pin.

## Results and evidence

The [aggregate qualification record](qualification/inkling-mini-hip-qualification.json) joins numerical, functional, artifact and resource evidence. The [raw HTTP proof](qualification/inkling-mini-hip/api.json), [native operator results](qualification/inkling-mini-hip/native-parity.txt), [vendor BLAS results](qualification/inkling-mini-hip/blas-parity.txt), server logs and resource envelopes are retained in the repository. The helper's raw result deliberately says `functional_probe_passed_not_numerically_qualified`; the aggregate adds the independently completed numerical checks.

- **Numerical:** 347 retained native checks passed against CPU/double references and exact fused/unfused assertions, without relaxing tolerances. FP32 relative-bias GEMM agreed with independent double sums to maximum absolute error `1.56211773e-6`; FP16-input dense GEMM to `6.14268356e-6`.
- **Real API:** streamed `ready`; streamed `lookup` with exact arguments `{"key":"redrock-check"}`; tool-result continuation returned `739281`. The model also answered correctly after a 7826-token prompt under the declared 8192-token context.
- **Limits:** an eight-token generation stopped with `finish_reason=length` and exactly eight generated tokens; a 9015-token prompt returned HTTP 400; a one-GiB expert-cache budget refused the exact model before `READY`, with no offload fallback. This is admission evidence, not deliberately induced VRAM exhaustion.
- **Recovery:** client disconnection produced engine `DONE ... cancel`; the next request succeeded. Two distinct server processes answered correctly, with owned VRAM allowed to return near baseline before restart.
- **Preservation:** 11 package tests, four qualification-helper tests, 14 CLI refusal cases, seven adapter protocol cases and the HTTP fake-engine suite passed. Existing Qwen/shared HIP code, frontend, llama.cpp gitlink and published release records are unchanged.

| Measurement | Observed result |
| --- | --- |
| Prompt processing | 7826 uncached tokens in 13.008315 s: **601.62 tokens/s** |
| Generation | 256 generated tokens, 255 timed decode steps in 3.262347 s: **78.16 tokens/s** |
| Startup to healthy API | **1.610 s / 1.608 s**, two fresh processes after hash verification with warm OS page cache |
| Peak sampled host process-tree RSS | **19,011,674,112 bytes (17.71 GiB)**; summed per-process high-water upper bound was the same in this run |
| Peak sampled whole-device VRAM | **20,448,362,496 bytes (19.04 GiB)**, including desktop allocations; initial baseline about 1.55 GiB |

At probe start the kernel reported 60.23 GiB host RAM, about 48.67 GiB available host RAM and 28.34 GiB available device VRAM; the resource envelopes retain the exact byte observations. The first generated token comes from prefill and is excluded from the timed decode-step rate. These are measured task/probe results, not a broad performance or quality benchmark. Memory was sampled every 50 ms: shared mappings may be double-counted in RSS, summed process high-water marks bound rather than exactly measure concurrent RSS, device totals include the desktop, and samples can miss transients. Cgroup charge peaks are retained separately and must not be substituted for host model RSS when file-cache pages were already charged elsewhere.

The human approved the exact 17.63-GB input, the running host, a 15-minute maximum per GPU attempt, 32 GiB process-group RAM, 28 GiB total-device VRAM and an 8 GiB available-host-RAM floor. Idle ComfyUI was temporarily paused with approval and restored after clean teardown. Each GPU run used only the selected render node and KFD, private PID/network namespaces, the existing runtime coordination lease, read-only repaired ROCr and kernel/resource monitoring. Passing native, BLAS and API envelopes recorded no new kernel events, preserved the desktop and released owned compute. Full local inputs/builds/evidence remain under `/mnt/storage/ai/redrock-issue8`; no active engine installation or host runtime was replaced.

## Immutable inputs and attribution

[The model manifest](qualification/inkling-mini-hip-inputs.json) selects `unsloth/Inkling-2-Mini-GGUF` revision `8078d149286eec7193795413b9e0643da6d11fa1`, file `Old_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file`: **17,629,759,456 bytes**, SHA-256 **`e62d463c135ffe283df8d94b279a96ca145ac128b52ce854478613045bf4de15`**. The complete downloaded file was verified before execution. Tokenizer and chat template are embedded in that exact hashed GGUF; retained adapter compatibility transformations are part of the identified adapter. There is no external tokenizer, drafter, projector, control vector or derived pack. Author/distributor cards and Apache-2.0 text were retained with the inputs; their [notice identities](qualification/inkling-mini-hip/model-notices.json) remain separate from the engine's MIT license.

The repository's root `Inkling-2-Mini-UD-Q4_K_XL.gguf` is a different mainline-layout artifact: 17,629,760,736 bytes, SHA-256 `e64276656f072e9b2197a61723667cbdb71da66a716ffd966e572c21088f3cd3`. It was not substituted or qualified. The deliberately selected retained-layout object is available at the immutable path above. Reassess metadata/layout and exact-input parity before selecting another object.

Project Maya was reassessed at maintained revision [`53885cef8faf0d3b3f7e28c0b1537a056c1f3ecf`](https://github.com/mw00/project-maya/tree/53885cef8faf0d3b3f7e28c0b1537a056c1f3ecf). Redrock selectively adapts its HIP fragment/packed-half helpers and 24-KiB RDNA prompt-attention/WMMA slice onto the retained Inkling engine, not its installer, model manager, Python server, offload lane or whole newer engine. Dense/batched expert products use existing hipBLAS rather than importing Maya's tuning store. `redrock.lock.json` and `third_party/project-maya/{NOTICE.md,LICENSE}` retain source identity and MIT attribution; llama.cpp remains at `db1f382bb25f9cc481893105dbead09c86a52f01`.

Real-device checks found and fixed two exactness problems: FP8 negative-zero loss through unnecessary float conversion, and a one-ULP fused MoE/convolution difference from cross-statement HIP FMA contraction. The original assertions remained intact and exhaustive scalar FP8 coverage was added. The first real-model request also exposed hipBLAS rejecting its advertised `COMPUTE_32F_PEDANTIC` selector. Inkling now requests supported full FP32 with default math, not XF32, independently tested against double products. Initial failures and fixes are recorded separately from the successful qualification; they were not kernel incidents or reasons to change the host kernel.

## Reconstruction

The qualified engine was built from clean Redrock commit **`8789c1d423f73c6ebe13a8582e6952d22baafd90`** using `tools/redrock.py`; the aggregate record retains the actual full source identity and flags. Use the aggregate record as authoritative if copying hashes. In a separate clean checkout with its pinned submodule prepared explicitly, create a fresh external candidate whose parent already exists:

```sh
python3 tools/redrock.py check
python3 tools/redrock.py build --engine strata-ik2 --backend hip --arch gfx1201 \
  --gpu-compiler /usr/lib64/rocm/llvm/bin/clang++ --jobs 2 \
  --candidate /chosen/existing-parent/inkling-mini-candidate
```

The compiler path is the observed host toolchain, not a portable assumption. Source, compiler versions, flags, environment and executable hashes are in `build.json`. Failures retain `INCOMPLETE`; retries use a fresh path. Build/serving perform no acquisition, installation or activation. Fresh build metadata intentionally remains unqualified; qualification attaches to the exact evidence-backed pair, not the target name. The [runtime inventory](qualification/inkling-mini-hip/runtime-inputs.json) identifies resolved engine/adapter dependencies and installed `gfx1201` rocBLAS data, with the [retained ROCr repair recipe](../releases/1.0.0.md#repaired-rocr); this is not a self-contained portable release.

Build the same-tree headless adapter separately:

```sh
cmake -S . -B /chosen/fresh-adapter-build -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF -DSTRATA_BUILD_IK2=OFF \
  -DSTRATA_FRONTEND=ON -DSTRATA_PORTABLE=ON \
  -DSTRATA_GGML_DIR="$PWD/third_party/llama.cpp" \
  -DGGML_CPU_KLEIDIAI=OFF -DGGML_OPENMP_FETCH=OFF \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON
cmake --build /chosen/fresh-adapter-build --parallel 2 --target unbiased-strata
```

Its executable is `/chosen/fresh-adapter-build/unbiased-strata`. Preserve its hash alongside the engine/input/runtime records. A later fresh build from `8d5a5f4e255ab2dfbe331584552c8ad9457f9f43` produced the byte-identical qualified engine and retained the complete qualification documentation; its [reconstruction metadata](qualification/inkling-mini-hip/reconstruction-build.json) is preserved. Arbitrary rebuilds are not promised to be bit-identical; changed artifacts or runtime/settings need affected verification rather than inheriting this result.

## Checks and reruns

Run `python3 -m unittest discover -s tools -p test_redrock.py -v`, the corresponding `test_qualify_inkling.py` suite, and `python3 tests/ik2_hip_cli_test.py /path/to/strata-ik2`. The recorded host uses `/usr/bin/python3` with `PATH=/usr/bin:/bin` to avoid pyenv fixture-environment injection. Existing adapter checks are `frontend/tests/engine_test.cpp` and `frontend/tests/e2e.sh`; their fake engine is not a substitute for the real-model proof.

For GPU numerical reruns, configure a separate external build from the candidate's recorded flags with `STRATA_BUILD_TESTS=ON`, changing output directories. Build `ik2_kernels_test`, `ik2_blas_test` and `strata-ik2`. Preserve reference tolerances and selected-device isolation. For an authorized real-model rerun:

```sh
python3 tools/qualify_inkling.py --attended \
  --candidate /path/to/candidate --adapter /path/to/adapter/unbiased-strata \
  --model /path/to/exact-selected-model.gguf \
  --manifest docs/qualification/inkling-mini-hip-inputs.json \
  --device GPU-26508202313fa2ac --pci 0000:03:00.0 \
  --runtime-library-path /path/to/verified-runtime-libraries \
  --output /chosen/fresh-evidence-directory
```

The helper validates input hashes, performs the real-model budget refusal, two startups, streamed text/tool exchange, near-limit context, a 256-token rate sample, actual output truncation, context refusal, cancellation/recovery and restart. It caps startup at 180 s and HTTP socket/stream completion waits at 120 s; it is not the resource supervisor. The [retained host-specific supervisor snapshot](qualification/inkling-mini-hip/host-supervisor.py) and envelope JSON show the actual isolation, read-only ROCr mapping, approved service lifecycle, cgroup limit, watchdog and kernel/resource observation used here. That snapshot describes this host, not a portable installer or a required Redrock serving dependency. Future downloads/GPU experiments still need their own selected-input/resource authorization.

## Limitations

The qualified boundary excludes other model files/quants, FP8 KV end-to-end claims, drafting, vision, multi-GPU, concurrent requests, very long context and other hardware. This is not 96K+ local coding-model qualification. No whole-model logit/KL equivalence or broad quality result is claimed from operator references or a few useful tasks. Tool arguments were naturally generated and successfully parsed; this engine does not enforce a JSON grammar or guarantee arbitrary `tool_choice` constraints. Existing stable release/installations remain separate from this completed issue-level qualification.
