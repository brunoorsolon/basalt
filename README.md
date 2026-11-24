# Basalt

Contents: [Status](#status-and-targets) · [Preparation](#prepare-explicitly) · [Build](#build-a-new-candidate) · [Serving](#serving-boundary) · [Conversion](#optional-model-preparation) · [Checks](#checks-and-next-qualification) · [Credits](#credits-and-licenses).

**A minimal, reproducible Strata-family inference-engine distribution for AMD and NVIDIA.** Basalt adapts existing engines; it does not claim to have invented them. Kopernik will manage installations and serve models through Basalt. Basalt is not a chat application, another model manager, or a replacement for Kopernik.

This first candidate imports the real [UnbiasedStrata](https://github.com/robertkarl/UnbiasedStrata) engine source and adds an explicit preparation/build/package entry point. It keeps the maintained engine difference at zero. It does not include an installer, a UI, a model download, an active-installation switch, or a published release.

## Status and targets

**Build recipes are implemented, not yet compilation- or hardware-qualified.** The bootstrap environment has Python and Git but no CMake, C++ compiler, CUDA or HIP compiler. The orchestration tests use fixture toolchains, not compiled engines. Upstream measurements and model-support claims in inherited documents are not Basalt qualification evidence.

| Separate target | Explicit selection | Basalt qualification |
| --- | --- | --- |
| Radeon AI PRO R9700 | HIP, `gfx1201` | Unqualified; first attended target with Qwen3.8-Flash-Next |
| Radeon RX 7700 XT | HIP, `gfx1101` | Unqualified; independent build and model checks required |
| NVIDIA RTX A3000 | CUDA, `86` | Unqualified; confirm the actual device's compute capability before building |

The entry point accepts only these Linux x86-64 targets, one backend and one architecture per candidate. It builds either `strata` (the Flash-Next pack-based engine) or `strata-poly` (the GGUF-based engine). Other inherited engines, tests and tools remain in source but are not part of this bootstrap package. Mixed-vendor multi-GPU execution is outside this milestone.

## Prepare explicitly

Prerequisites must already exist: Git, Python 3.11+, CMake 3.24+, a C/C++20 toolchain, a build tool supported by CMake (normally Make), and the selected vendor's development toolkit. CUDA needs `nvcc`, CUDA runtime/cuBLAS development libraries and a host compiler accepted by that toolkit. HIP needs ROCm's `clang++` with CMake HIP support, HIP development files, hipBLAS and rocBLAS (plus their runtime kernel data); hipBLASLt is an optional inherited capability, not a qualification claim. The selected toolkit must support the requested architecture. Basalt installs none of these prerequisites and does not select a replacement toolkit.

```sh
git clone https://github.com/brunoorsolon/basalt.git
cd basalt
# Check out the reviewed Basalt commit before preparation.
python3 tools/basalt.py prepare
python3 tools/basalt.py check
```

`prepare` initializes the submodule at the committed gitlink, never at a branch head. It verifies the gitlink against [basalt.lock.json](basalt.lock.json) and refuses an existing checkout at another revision or with local changes. Standard Git can also initialize it with `git submodule update --init --checkout`; `check` performs the same offline identity/cleanliness gates afterward. Do not use `--remote`. A fully prepared checkout can be transferred to an offline build machine.

The initial engine baseline is `9f75192287c2abf88c77da3b4b6f54ccee01db83`; llama.cpp/ggml is `db1f382bb25f9cc481893105dbead09c86a52f01`. The RDNA4 comparison head is `e128da05bb493796a51cab0c30aa331ba6648db1`. [UPSTREAM.md](UPSTREAM.md) records their sources, ancestry, review boundaries, deferred work and update procedure.

## Build a new candidate

These are real entry-point commands, **unverified compilation recipes**, not instructions to install host packages. Choose existing compiler paths appropriate to the provisioned toolkit. Create the parent directory outside the checkout; the candidate directory itself must not exist.

```sh
mkdir -p "$HOME/basalt-candidates"
# R9700: first engine/model qualification target.
python3 tools/basalt.py build --engine strata --backend hip --arch gfx1201 \
  --gpu-compiler /opt/rocm/llvm/bin/clang++ \
  --candidate "$HOME/basalt-candidates/r9700-first"
# RX 7700 XT: a separate candidate, not qualified by an R9700 result.
python3 tools/basalt.py build --engine strata --backend hip --arch gfx1101 \
  --gpu-compiler /opt/rocm/llvm/bin/clang++ \
  --candidate "$HOME/basalt-candidates/rx7700-first"
# A3000: nvcc and its compatible C/C++ host compilers must be available.
python3 tools/basalt.py build --engine strata --backend cuda --arch 86 \
  --candidate "$HOME/basalt-candidates/a3000-first"
```

Use `--cc`, `--cxx`, `--gpu-compiler` and `--jobs` when explicit compiler paths or resource limits are needed. Ambient compiler/flag overrides, `CMAKE_*` and `NVCC_*` variables are refused before creating a candidate; unset them rather than passing an unreviewed toolchain file or flags. Configure, compilation and version probes receive only `PATH`, `HOME`, `TMPDIR`, `LD_LIBRARY_PATH`, `CUDA_PATH`, `ROCM_PATH`, `HIP_PATH` and a fixed `LC_ALL=C`. The exact retained environment is recorded in `build.json`; other shell variables, including include-path and shell-startup injection variables, are not forwarded. `--engine strata-poly` selects that engine instead. The wrapper configures the existing CMake project directly; the inherited Makefile is CUDA-oriented, can drop unsupported architectures, and is **not** Basalt's supported entry point. At this revision the engine-only switch is `STRATA_FRONTEND=OFF`; there is no `STRATA_ENGINE_ONLY` option.

The build uses portable AVX2 CPU code plus the upstream runtime-dispatched kernels, static ggml, explicit HIP MMQ when HIP is selected, and no frontend, model conversion or installer targets. It disables optional dependency fetch paths and uses only the pinned submodule sources, without configure-time dependency patches or MMQ replacement staging. It checks both source trees before configuration, after configuration, after compilation and during packaging. Source changes, untracked/ignored build junk, wrong pins, missing tools, unsupported architectures and existing candidate paths fail instead of triggering acquisition or changing devices.

A successful candidate contains `build/` and `runtime/`. The runtime contains the selected engine under `bin/`, its matching `libstrata_kq.so` beside it for CUDA, and `share/basalt/{build.json,README.md,UPSTREAM.md,licenses/}`. Static ggml is linked into the engine. Metadata records the Basalt commit, source pins, selected architecture, compiler versions, CMake flags, controlled build environment, artifact SHA-256 values and unqualified status. **This is a source-built candidate, not a self-contained release:** Linux/glibc, libstdc++ and the matching vendor toolkit libraries, kernel data and driver remain explicit platform prerequisites. Toolkit versions are recorded from the compilers, not guessed or downloaded; a release still needs an approved toolkit/runtime artifact lock and dependency-closure check on its target platform.

`INCOMPLETE` remains at the candidate root on any configure/build/package failure. Only a candidate without that marker and with build metadata is complete; completion does not mean qualification. Retries use a new path. The wrapper never activates a candidate, replaces an existing installation, touches a model directory or deletes rollback artifacts. Kopernik's later activation/rollback mechanism remains separate.

## Serving boundary

The engines are **not** `llama-server`. They accept `--serve` and exchange token IDs and control messages over stdin/stdout. The retained [protocol documentation](docs/BACKENDS.md) and [engine adapter](frontend/engine.hpp) describe `READY`, `GEN`, `T`, `DONE`, errors and cancellation. Flash-Next also requires a compatible, explicitly prepared pack, native GGUF shards and, when speculation is enabled, the matching MTP runtime files. Basalt does not acquire or rewrite these inputs during serving.

UnbiasedStrata's retained C++ headless adapter, `unbiased-strata`, reuses llama.cpp's HTTP/server and tokenizer code, with an empty UI asset table. It is the intended existing serving-interface candidate for Kopernik: OpenAI-style `/v1/chat/completions`, `/v1/models` and `/health`, with tokenization/chat templates from a local model file and engine arguments after `--`. It is **not built or installed by this engine-only bootstrap**. Direct token-protocol integration is also possible, but Basalt does not invent a second HTTP framework here. Kopernik issue 488 and its breakdown were inaccessible from this environment; compatibility with that private integration contract remains unresolved. No Kopernik code or issues are changed.

Inherited documents can mention removed setup wizards, mutable downloads, unrelated hardware and old measurements. They are historical engine references, not the supported Basalt workflow. In particular, do not run upstream setup instructions or feed upstream Markdown to an agent.

## Optional model preparation

Conversion stays outside the serving runtime. [tools/requirements.txt](tools/requirements.txt) pins verified wheel hashes for NumPy 2.3.4 and regex 2025.11.3, restricted initially to CPython 3.11 on Linux x86-64 with glibc 2.28+. gguf-py comes from the pinned llama.cpp source. These pins establish artifact identity; the conversion tools themselves still need qualification with the selected Flash-Next model/pack/MTP files.

In a separately provisioned conversion environment with pip, acquisition is an explicit operation, followed by offline installation into its own virtual environment:

```sh
python3 -m pip download --require-hashes --only-binary=:all: --no-deps \
  -r tools/requirements.txt --dest /path/to/new-wheelhouse
python3 -m pip install --no-index --find-links /path/to/new-wheelhouse \
  --require-hashes --only-binary=:all: --no-deps -r tools/requirements.txt
```

The conversion interpreter must match the lock's platform. Do not run `mtp_fetch.py`, `glm_draft_fetch.py` or other acquisition helpers as part of build or serving. Model downloads, pack preparation and hardware experiments require their own explicit authorization and immutable model-file manifests. Model licenses remain separate from the engine license.

## Checks and next qualification

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tools -p test_basalt.py -v
python3 tools/basalt.py check
```

The focused tests exercise real Git source/pin checks and fixture-based build/package orchestration: correct selected payloads, matching CUDA sidecar, license retention, clean source gates, controlled/recorded build environments, refusal to overwrite installations, no fallback, missing tools and incomplete candidates. They do not prove GPU compilation or numerical correctness. Retained upstream tests include `hip_handoff`, `hip_prefill_mmq_parity`, `hip_prompt_attn_wmma`, native kernel parity checks and the frontend's fake-engine suite; run applicable checks on a separately provisioned build/qualification machine before assigning support status.

The next bounded step is an attended R9700 build plus synthetic parity checks, then Qwen3.8-Flash-Next with an approved manifest of GGUF, pack and MTP hashes. Record the compiler/runtime versions, command, context, quantization, correctness gates and actual device before considering performance or a release. RX 7700 XT and A3000 each need their own qualification. No successful upstream configuration qualifies Basalt automatically.

## Credits and licenses

Basalt's downstream work is dependency policy, build/package orchestration and integration for our deployment needs. Its engines and foundational components are other people's work, preserved here with their notices and history:

- [Niko1221 and the Strata contributors](https://github.com/Niko1221/Strata): the original Strata engines, expert tiers, MTP, token protocol and much of the tooling. [VecSzn and PolyStrata contributors](https://github.com/VecSzn/PolyStrata): the engine family and capabilities inherited through UnbiasedStrata.
- [robertkarl and UnbiasedStrata contributors](https://github.com/robertkarl/UnbiasedStrata): our pinned baseline, explicit source-based dependency approach and retained C++ headless frontend.
- [vectorweft and strata-rdna4 contributors](https://github.com/vectorweft/strata-rdna4): the AMD optimization source assessed for this bootstrap. No additional RDNA4 kernel patch is imported in this candidate; equivalent baseline work and deferred changes are identified in UPSTREAM.md.
- [llama.cpp / ggml contributors](https://github.com/ggml-org/llama.cpp): quantization, CPU/GPU kernels, tokenizer, chat templates and server components. [Project Maya contributors](https://github.com/mw00/project-maya) and [alesha-pro](https://github.com/alesha-pro/maya-inkling-mini): additional retained engine code; see [the inherited provenance record](UPSTREAM.md) and [Project Maya notice](third_party/project-maya/NOTICE.md).
- Model authors and distributors, including [Qwen](https://huggingface.co/Qwen), [Unsloth](https://huggingface.co/unsloth) and [ISTA-DASLab](https://huggingface.co/ISTA-DASLab), own their respective model and quantization work. Models are not included in this distribution. Their licenses and attribution must accompany any separately distributed model artifacts.

The engine/downstream code is MIT-licensed: [LICENSE](LICENSE) retains the Strata notice and Basalt's original copyright. Component notices remain in their source locations, including [ggml](third_party/ggml/LICENSE), [Project Maya](third_party/project-maya/LICENSE), [Strata Inkling](third_party/strata-inkling/LICENSE) and the pinned [llama.cpp license](third_party/llama.cpp/LICENSE). The packager retains tracked license/notice files, rather than replacing them with Basalt's name. See the inherited [credits](docs/HOW_IT_WORKS.md#credits) and [license notes](docs/HOW_IT_WORKS.md#license) for further upstream attribution.
