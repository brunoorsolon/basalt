# Inkling 2 Mini / R9700 HIP candidate — issue #8

**Status: software candidate, not qualified support. Issue #8 remains open until selected-device operator parity and the real-model API proof pass.** This work does not change the accepted Qwen 1.0.0 pair, dependency pins, runtime repair, release tags, installations or model directories. The new candidate is selected explicitly; existing engines keep their defaults.

## Selected boundary and source assessment

On 2026-11-26 Project Maya's maintained `main` resolved to [`53885cef8faf0d3b3f7e28c0b1537a056c1f3ecf`](https://github.com/mw00/project-maya/tree/53885cef8faf0d3b3f7e28c0b1537a056c1f3ecf). Its `docs/INKLING2.md` section 8 documents Mini (26B, 24 physical layers and 48 visits), Linux HIP, a resident configuration, RDNA prompt attention and vendor BLAS products. Its measured AMD example is a Radeon 8065S, **not this R9700**. Its NVIDIA speed and context measurements are not Redrock results.

The smallest coherent adaptation is the retained `src/ik2` engine plus Maya's HIP fragment/packed-half primitives and the 24-KiB RDNA prompt-attention/WMMA slice. Dense and batched expert products use Redrock's existing hipBLAS interface, rather than importing Maya's hipBLASLt tuning store and its dependencies. Decode experts default to direct row dots on HIP; the CUDA-fragment emulation remains available to the operator checks. The existing scalar CPU reference in `tests/ik2_kernels_test.cu` covers normalization, convolution, residual attention, routing, quantized decodes, experts, attention and token-path operations. Compilation of that test is not numerical parity.

The selective source identity is in `redrock.lock.json`; attribution and MIT permission remain in `third_party/project-maya/{NOTICE.md,LICENSE}` and are packaged. The new files are `src/ik2/ik2_rdna_attention.inc`, `ik2_rdna_wmma.hpp` and an Inkling-local HIP API shim. The qualified Qwen runtime shim is unchanged. No installer, Python server, model manager, mutable source fetch, whole-engine replacement, newer offload lane, vision or drafting changes are imported. The llama.cpp gitlink remains `db1f382bb25f9cc481893105dbead09c86a52f01`.

The initial candidate deliberately admits Linux x86-64, HIP `gfx1201`, one isolated discrete GPU, the exact Mini loop geometry, resident experts and a bounded text-only configuration. Defaults are context **8192**, prompt chunk **256**, FP16 KV, no drafter. HIP startup rejects context outside 1–8192, chunk outside 1–256, extra GPUs, Small/other geometries, pictures, drafting, the legacy `--serve-ik2` mode and the managed/CPU-offload modes. Failure to retain the resident path is fatal: there is no silent host-offload/device fallback. This is an inference integration boundary, **not qualification of a coding model at 96K+**.

## Hardware feasibility, not execution evidence

Read-only host observations on 2026-11-26 at approximately 22:51 UTC:

| Item | Observed |
| --- | --- |
| Kernel | `7.3.4-200.fc44.x86_64` |
| Target PCI device | `0000:03:00.0`, AMD `1002:7551`, Radeon AI PRO R9700 |
| Kernel-reported VRAM total | 32,061,259,776 bytes (29.86 GiB usable address space reported by sysfs) |
| Kernel-reported VRAM used | 1,157,885,952 bytes; approximately 28.78 GiB free at that instant |
| Host RAM total | 63,151,880 KiB (60.23 GiB) |
| Host RAM available | 55,099,500 KiB (52.55 GiB) |
| HIP compiler | `/usr/lib64/rocm/llvm/bin/clang++`, `clang version 20.0.0.rocm` |

Maya's layout estimate is about 14.7 GiB of experts plus 1.1 GiB of dense blocks. At 8192, FP16 global KV is approximately 256 MiB; sliding KV is approximately 200 MiB with the selected chunk, before workspace, snapshots, graphs, allocator overhead and the engine's reserve. That gives a plausible initial memory budget below the observed free VRAM; **it is not an allocation or peak-memory measurement**. Startup still checks actual selected-device free memory and refuses a nonresident configuration. Host free memory also changes; these figures are not a reservation.

Qualification targets the running host kernel, 7.3.4. Historical release results do not impose a kernel downgrade or an additional standing gate. The qualified isolated-device setup requires the repaired ROCr described in [the release record](../releases/1.0.0.md#repaired-rocr). Do not weaken isolation to work around a ROCr failure. Model acquisition and attended execution need explicit authorization for the selected inputs and limits; an unattended implementation request did not select or authorize those downloads/experiments. Consequently the new candidate has no GPU/model qualification result.

## Exact model and tokenizer/template inputs

The proposed first attended input is [the immutable manifest](qualification/inkling-mini-hip-inputs.json), resolved anew from Hugging Face metadata, not implicitly downloaded. It selects:

- Repository `unsloth/Inkling-2-Mini-GGUF`, revision `8078d149286eec7193795413b9e0643da6d11fa1`.
- File `Old_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file`, **17,629,759,456 bytes**, SHA-256 `e62d463c135ffe283df8d94b279a96ca145ac128b52ce854478613045bf4de15`.
- Author metadata `thinkingmachines/Inkling-2-Mini` at `c9d028302cd0f035fdf8e7e91f089a4ce94d2195`, Apache-2.0, `Inkling2LoopForConditionalGeneration` / `inkling2_loop`.
- Tokenizer and template embedded in that exact GGUF, immutably covered by its full-file hash. Adapter transformations are the retained compact-JSON Jinja compatibility, tool-argument canonicalization and Inkling typed-output parser in `frontend/strata_context.cpp` / `frontend/inkling.hpp` at the recorded source revision.
- No external tokenizer, template, vision projector, drafter, control vector or derived pack. No model license is replaced by the engine's MIT license; retain the model/distributor's Apache-2.0 notices with any separately acquired/distributed weights.

**Changed upstream assumption:** the maintained repository's root `Inkling-2-Mini-UD-Q4_K_XL.gguf` is now **17,629,760,736 bytes**, SHA-256 `e64276656f072e9b2197a61723667cbdb71da66a716ffd966e572c21088f3cd3`. It is not the old artifact used by Maya's inspected documentation. The retained-layout object is deliberately selected for the retained engine; it is still available at the named immutable revision/path. Do not rename/substitute the current root object or inherit compatibility from a similar filename. Reassess the conversion/metadata change with exact-input parity before selecting that other object. Metadata observations are not a full-file hash verification or model-load test; the attended probe verifies the bytes before starting a process.

## Reconstruct a fresh candidate

Use a clean checkout at the desired reviewed commit with its pinned llama.cpp submodule prepared explicitly. The builder never downloads models or dependencies during configuration/serving. Provisioning and source acquisition follow host remote-access policy; no system updates are part of this recipe. The parent directory below must exist; the candidate itself must not exist.

```sh
python3 tools/redrock.py check
python3 tools/redrock.py build --engine strata-ik2 --backend hip --arch gfx1201 \
  --gpu-compiler /usr/lib64/rocm/llvm/bin/clang++ --jobs 2 \
  --candidate /chosen/existing-parent/inkling-mini-candidate
```

The compiler path above is the observed host toolchain, not a portable assumption. Select an already provisioned ROCm compiler explicitly on another host. Candidate metadata contains source pins, compiler versions, exact CMake flags/environment and executable hashes. `INCOMPLETE` is retained on failure; retry at a fresh path. Model inputs and active installations are untouched. The runtime includes the engine, source licenses, this guide and the immutable proposed model manifest. Linux, vendor libraries/kernel data, driver and repaired ROCr remain explicit prerequisites, not a newly bundled runtime closure.

Build the **same-tree headless adapter separately**, as for the Qwen baseline; do not use Maya's Python server:

```sh
cmake -S . -B /chosen/fresh-adapter-build -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF -DSTRATA_BUILD_IK2=OFF \
  -DSTRATA_FRONTEND=ON -DSTRATA_PORTABLE=ON \
  -DSTRATA_GGML_DIR="$PWD/third_party/llama.cpp" \
  -DGGML_CPU_KLEIDIAI=OFF -DGGML_OPENMP_FETCH=OFF \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON
cmake --build /chosen/fresh-adapter-build --parallel 2 --target unbiased-strata
```

The executable is `/chosen/fresh-adapter-build/unbiased-strata`. There is no installation/activation step. Preserve the adapter's hash with the engine's `build.json` and the platform runtime closure. New candidates remain unqualified regardless of successful compilation.

## Checks and attended gates

Offline software checks:

```sh
PATH=/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -m unittest discover -s tools -p test_redrock.py -v
PATH=/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -m unittest discover -s tools -p test_qualify_inkling.py -v
python3 tests/ik2_hip_cli_test.py /path/to/candidate/runtime/bin/strata-ik2
c++ -std=c++20 -pthread -I frontend frontend/tests/engine_test.cpp -o /tmp/redrock-engine-test
/tmp/redrock-engine-test frontend/tests/fake-engine.py
frontend/tests/e2e.sh /path/to/adapter/unbiased-strata
```

The explicit system Python/PATH avoids a host pyenv shim injecting extra variables into the fixture toolchain's environment. The HTTP fake-engine suite uses a vocabulary-only GGUF retained in the pinned dependency, not Inkling weights; it cannot qualify Inkling output or its tokenizer.

For numerical parity, configure a **separate external build** with the candidate's exact recorded `cmake_flags`, changing output directories and `STRATA_BUILD_TESTS=ON`. Build `ik2_kernels_test` and `strata-ik2`. That configuration also registers `ik2_hip_cli_test`. Run the GPU test only after selecting/authorizing the exact isolated R9700, runtime and resource envelope. Compare against the existing CPU/double references without relaxing tolerances to make HIP pass. A failing operator is a stop condition, not permission to try other devices or broaden the port.

Once acquisition of the manifest's exact 17.63-GB object and a bounded attended run have been authorized, stage the model **outside** source/build/serving. Verify its size/hash and retain its model notices. With the selected device's immutable ROCr UUID and repaired runtime path, run:

```sh
python3 tools/qualify_inkling.py --attended \
  --candidate /path/to/candidate --adapter /path/to/adapter/unbiased-strata \
  --model /path/to/exact-selected-model.gguf \
  --manifest docs/qualification/inkling-mini-hip-inputs.json \
  --device GPU-SELECTED-R9700-UUID --pci 0000:03:00.0 \
  --runtime-library-path /path/to/verified-rocr-and-toolkit-libraries \
  --output /chosen/fresh-evidence-directory
```

This operation performs no acquisition. It checks candidate and model hashes before launching, selects `ROCR_VISIBLE_DEVICES=<UUID>` with HIP ordinal 0, rejects additional visible devices in the engine, and binds the adapter only to loopback. It launches one server/request at a time, with fixed 8192/256, FP16 KV and no drafter. It tests streamed text, a streamed `lookup` call with exact JSON arguments, a real tool-result continuation, an eight-token output budget, an oversized-context refusal, disconnect/cancellation acknowledged by the engine, a clean next request, termination and a fresh-server request. Both process groups are bounded to this probe and cleaned up. Startup is capped at 180 seconds; individual HTTP operations use a 120-second socket timeout. Observe the operator's resource limits too; these socket timeouts are not a whole-run watchdog.

`evidence.json` and raw server logs retain exact engine/adapter/model identities, runtime path, request outcomes, two startup times, prompt/generation timing rates with reused prompt tokens excluded, and 50-ms sampled process-tree RSS/device-wide VRAM. RSS can double-count shared mappings, VRAM includes desktop/other allocations and sampling can miss transients: these are explicit measurement limitations, not allocator-level peak guarantees. Retain the independently verified runtime-library closure with the evidence. The probe returns `functional_probe_passed_not_numerically_qualified`, never “qualified”; operator parity and useful/quality outcomes remain separate gates.

## Delivery evidence and remaining acceptance criteria

[The retained build/evidence record](qualification/inkling-mini-hip-build.json) contains the actual compiled source commit, compiler/environment/CMake inputs, engine/adapter/operator-test hashes and check outcomes. The compiled source is `3c572df4bd6dbc8ba946b65e7707f546d29fa61c`; later evidence-only commits do not change that source identity. Binaries and raw logs under `/tmp` are scratch, not durable release assets. Reconstruction uses the recorded clean source and flags, not those temporary paths.

Actual software observations from this implementation:

- The R9700 HIP engine compiled and packaged successfully with the supported `tools/redrock.py` entry point, outside the checkout, using the observed ROCm compiler. Earlier incomplete attempts were retained; fixes are recorded in branch history rather than hidden source substitutions.
- The retained operator-reference executable compiled for `gfx1201`; compilation required using the C++ host linker for this HIP-only test's C++ runtime libraries. Its numerical assertions have not been evaluated on the device.
- Eleven build/package orchestration tests passed, including the explicit Inkling selection, wrong backend/architecture refusal, source/overwrite gates, Maya notices, input guide and unqualified metadata.
- Four offline qualification-helper tests passed: exact input identities/incomplete refusal, SSE text/tool fragments and tool-result forwarding, truncated-stream refusal, isolated process-group cleanup and uncached prompt/decode rate calculations.
- Fourteen pre-GPU CLI refusal cases passed, including bounded arguments, forbidden offload settings and a minimal wrong-family GGUF fixture. Seven real adapter/fake-engine protocol cases and the existing HTTP fake-engine suite passed, including limits, interruption/recovery, UTF-8 streaming and shutdown. The adapter, literal, picture-cache and Hy4 sampling executables also built; these are affected-existing-surface checks, not real Inkling generation.

The implementation leaves the acceptance checklist **incomplete**: real selected-artifact loading, HIP/CPU numerical parity, useful text/tool results through the actual model, real context/memory-pressure refusals, measured prompt/generation/startup rates and peak-resource evidence, and affected Qwen hardware requalification remain pending. The selected weights were not acquired; no unattended GPU/model result or inherited support claim is fabricated. Stop at those gates and preserve the existing qualified installation until the human authorizes the named inputs and attended limits.
