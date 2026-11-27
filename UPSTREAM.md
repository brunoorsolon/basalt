# Redrock upstream record

Contents: [Baselines](#redrock-baselines-and-review-boundaries) · [RDNA4 assessment](#initial-rdna4-dispositions) · [Updates](#repeatable-update-procedure) · [Inherited sources](#inherited-unbiasedstrata-provenance) · [Inherited engine changes](#engine-source-src-include) · [Inherited removals](#removed).

## Redrock baselines and review boundaries

Redrock owns its downstream changes and eventual releases. Upstream authors own their contributions; their notices and ancestry remain intact. Source identity comes from the Redrock release commit and its dependency pins, not a review bookmark. The accepted [1.0.0 baseline](releases/1.0.0.md) qualifies a bounded R9700 configuration; publication follows [RELEASING.md](RELEASING.md).

| Source | Canonical URL / tracked ref | Frozen initial head |
| --- | --- | --- |
| Engine baseline | https://github.com/robertkarl/UnbiasedStrata / `refs/heads/main` | `9f75192287c2abf88c77da3b4b6f54ccee01db83` |
| AMD comparison | https://github.com/vectorweft/strata-rdna4 / `refs/heads/main` | `e128da05bb493796a51cab0c30aa331ba6648db1` |
| Actual llama.cpp/ggml dependency | https://github.com/robertkarl/llama.cpp.git / committed gitlink, not a tracked branch | `db1f382bb25f9cc481893105dbead09c86a52f01` |

The project was named Basalt at import. Import method: merge UnbiasedStrata's unrelated history into Basalt's feature branch, retaining Basalt's initial commit `a0d0f20159d9cc2e551acf3aec792bf954c2cd91`. The import commit is `02d73945c4e8221aced960b119b35ecb546e4a94`; both parents are retained. No `.git` replacement, origin change, history rewrite or RDNA4-tree merge was used. The Basalt license notice was restored alongside the retained Strata notice in the downstream commit. The imported baseline left `src/`, `include/` and the upstream CMake build unchanged; the bounded Redrock engine fixes below are additional downstream changes.

The inherited record below is retained as historical provenance, not silently rewritten as Redrock validation. In particular, its earlier submodule versions describe earlier imports: the actual current dependency is the gitlink/lock above. Vendored components are frozen by the Redrock commit's tree; their upstream origins remain documented below. The selected entry point builds `strata`, `strata-poly` or the explicitly selected R9700/HIP `strata-ik2` candidate; it neither builds the other vendored engines nor fetches their prerequisites.

### Redrock engine fixes

- **HIP EXPDEC shared alignment:** in `src/kernels/cuda/iq_kernels.cu`, remove redundant declaration-level `alignas(16)` from the gate/up and down `int4` shared-memory arrays inherited from UnbiasedStrata `9f75192287c2abf88c77da3b4b6f54ccee01db83`. ROCm Clang rejects that attribute placement. CUDA/HIP `int4` already provides the required alignment; a compile-time assertion preserves the 16-byte layout requirement. Array dimensions, staging/reduction code, dispatch, numerical operations and dependency pins are unchanged. This is a local adaptation of the canonical Redrock baseline, not an import from the deferred RDNA4 range. Compilation and hardware parity remain distinct evidence gates.

### Inkling Mini HIP selective adaptation (issue #8)

On 2026-11-26, maintained Project Maya `main` resolved to `53885cef8faf0d3b3f7e28c0b1537a056c1f3ecf`. Redrock adapts only its HIP fragment/packed-half helpers and 24-KiB RDNA prompt-attention/WMMA slice onto the retained Inkling engine. Dense and batched expert products use the existing hipBLAS contract; decode experts default to direct row dots on HIP. The Maya installer, serving stack, newer offload/thread lanes, vision and drafting changes are not imported. `redrock.lock.json` records this bounded source identity; neither engine-feed review bookmark nor llama.cpp pin advances. The existing Project Maya MIT notice remains packaged.

[The Inkling qualification record](docs/INKLING_MINI_HIP.md) now preserves completed selected-device operator and real-model API evidence on 2026-11-27, the deliberately selected immutable model input, reconstruction and measured resource/performance limits. The exact R9700/HIP pair passed 347 native checks, independent FP32/FP16-input BLAS comparisons and streamed text/tool-result, limit, cancellation and clean-restart checks at 8192 context. This result does not alter the 1.0.0 stable release boundary.

Real-device downstream corrections preserve unit-scale FP8 signed zeros and restrict HIP fused-MoE convolution contraction to statement-local FMA, keeping original exactness assertions intact. The advertised hipBLAS pedantic compute enum returned status 7; Inkling's local shim now uses supported full FP32/default math, independently checked against CPU double sums (not XF32). No shared Qwen HIP shim, dependency pin or host runtime was changed.

### Review bookmarks

- **UnbiasedStrata:** initial snapshot accepted at `9f75192287c2abf88c77da3b4b6f54ccee01db83` for bootstrap source import, licensing, dependency/build mechanisms and headless-serving interface. This is the initial comparison boundary, not a retrospective kernel audit or an assertion that all inherited engines work. Future bounded updates start here.
- **strata-rdna4:** last fully reviewed contiguous change after the selected development base is `7908212f1d06536d10a5077049d4cbb212beaee1` (the gfx1201 dispatch/dot4/doorbell change). It is already present or superseded in the newer baseline. The development base is `a79080535d1b2a71a3419a0d97d8e7dca194b0f1`, the first RDNA4 commit's direct parent, identified from Git rather than a version-name guess.
- The remaining first-parent range `7908212f1d06536d10a5077049d4cbb212beaee1..e128da05bb493796a51cab0c30aa331ba6648db1` has an initial disposition assessment below, **not a fully reviewed-through bookmark**. Its correctness-sensitive ports and the large intervening merge remain deferred. Fetching, reading commit subjects or comparing snapshots alone does not advance the bookmark.

`git merge-base` found no common ancestor between the two frozen heads. UnbiasedStrata imported PolyStrata as snapshots (including its `0197ae86ebc91b8a8735700dde9a6dbd667cea28` import), while RDNA4 retains Strata's history and later merged other upstream work. Therefore a whole-tree diff is not an RDNA4 patch series. The initial inventory is the 41 first-parent commits in `a79080535d1b2a71a3419a0d97d8e7dca194b0f1..e128da05bb493796a51cab0c30aa331ba6648db1`; the broad merge in that interval is classified separately below. No convenient common merge base is invented.

## Initial RDNA4 dispositions

**Accepted additional RDNA4 code: none.** Preserve the newer engine, AMD architecture dispatch and CUDA behavior first. The initial AMD integration is the inherited HIP backend with explicit `gfx1201`/`gfx1101` build selection and the baseline's direct compilation of pinned ggml sources. The observations below are static source/build evidence, not numerical or hardware validation. Full source hashes identify revisit work even when no code was imported.

| Family / source commits | Disposition, evidence and revisit condition |
| --- | --- |
| Architecture dispatch, signed dot4, host-visible doorbell: `7908212f1d06536d10a5077049d4cbb212beaee1` | **Already present / superseded.** `cmake/hip_backend.cmake` validates multiple AMD architectures; `src/core/device.cu` compares the device against `STRATA_HIP_ARCHS`; `include/strata/hip_compat/intrinsics.hpp` includes gfx1201 signed dot4. `src/kernels/cuda/elementwise.cu` uses volatile doorbells and HIP post-store system fences, rather than RDNA4's atomic-store adaptation. Retain these newer guards and the existing `hip_handoff` check. No duplicate cherry-pick. |
| Prompt-attention WMMA: `7f4a6375cd152fc15607d1c682dff8ae9824b5c2` | **Already present / evolved.** `src/kernels/cuda/qsa_prompt_attn.cu` already contains gfx12 WMMA, architecture-aware builtin dispatch and a gfx11 path, with `hip_prompt_attn_wmma` in CMake. The old gfx12-only implementation is not a reason to overwrite it; numerical equivalence remains a target-hardware check. |
| Native K-quant experts and Unsloth packing: `e88421345b8bdd4691aeda7805d7559bd9fb8452`, `6002e2fbfd1502472155cc3b48ead8a65ca47f4a` | **Capability already present; exact port deferred.** The baseline has mixed-format experts, Q4_K/Q5_K/Q5_1/Q6_K prompt instances, native projections and pack/MTP tools. That is not proof of identical layouts or quantization. Revisit only with an approved exact GGUF/pack/MTP manifest and parity evidence exposing a missing case; do not substitute old tools over newer layouts. |
| Dense Q8_0 WMMA: `98ce2cbe707c6668ac0d28c2975a7454e91dde8d`; long-row MMVQ: `710b7f24d01f7bf2982e178b76d1cd9cd942e387` | **Deferred.** Dense WMMA changes FP16 rounding and unconditionally uses a gfx12 builtin inside the HIP compilation branch; its support check tests shape, not device architecture. A direct port can break the gfx1101 build. The MMVQ change alters the long-row reduction/warp layout. Revisit each independently after R9700 baseline qualification, with architecture guards, connected projection callers, synthetic parity and accuracy measurements; preserve gfx1101 and CUDA builds. |
| Pointer-table expert MMQ: `1ef49d5d23e74f0d06245100c4a671fa00ecadeb`; source staging: `749e427920bfbefba9b9da2f29f9dc666269f6e7` | **Optimization deferred; staging mechanism rejected.** RDNA4 adds `mmq_args.x_ptrs`, caller `Product.w_ptrs`, and stages a customized `third_party/ggml-mmq/mmq.cuh` together with copies of the dependency's relative-include translation units. Its mismatch check warns instead of rejecting. The baseline's HIP path compiles the submodule's untouched quantizer/MMQ instances directly; its optional CUDA pointer-table scheduler lives in separately named Strata source and is fingerprint-gated. Keep that baseline. Revisit HIP pointer tables only as a reviewed commit in a pinned llama.cpp fork plus coherent caller/build changes and tail/bounds/parity tests; never configure-time replacement files or copied dependency implementations. No dependency modification is required for this candidate. |
| MTP head quantization/allocation and second draft vocabulary: `50559529b522c97bc65ba73b8f8d17a63820c602`, `a8caf8996d200dabffc4f496e9cbd6a93608a425`, `51c94ea2c1bab0a37b40c4f54e08edc07a1c1c5e`; related portion of `558aecbea9f7e45dbcc4a24cb2e179adc657868a` | **Deferred.** These changes span native head packing, chunked device re-quantization, cache VRAM reservation, graph capture and `draft_vocab_steps.bin`. The baseline already has Q4-head machinery but is not established equivalent to this chain. The additional vocabulary/pack identities and acceptance/correctness effects are unqualified. Revisit as one cohesive model/engine contract with the exact MTP inputs, memory-pressure tests and speculative-vs-target correctness gates. Do not copy the fork's binary profiles into the package without that evidence. |
| Fused/shared-expert/hyper-connection/GDN/router/verify kernels and CPU quant kernels: `26327dd280ddf32f27cc279a2b0c320da073389f`, `89530bacd91cd8ea42987259df45c5e12ad90491`, `80126b1b9517be5810e5db21ad89dd0e3b8624bc`, `07f6067b3d5240db80a7e0ce7a7c9acec5bcbf3f`, `45e8c73a41a6430351d428f276e1411f7d382eb3`, `d066fe0d689d82a2ef5f02c7c3738a746e38448d`, `fa0dcaaef6d6ab51644a45e9bcfa839951065f0e`, `86338b5dccefff2b2a7ef2ea925ee13131c44638`, `27dac2e94c265d48196e9319466b17f726839135`, `70e4a392a93f858c6ab4ecf363783b0158e97da3`, `d064f28913d9bbcdadcd4f5059bcb5571dcc01b2`, `a4d0af68df56bb42bcd2a9993e22cdb1ec17a41b` | **Deferred, not presumed missing.** The newer baseline has independently evolved fused kernels, ISA/runtime dispatch and expert layouts. A snapshot difference does not establish a portable optimization; these changes reach session scratch sizes, model geometry, floating-point reductions and CPU/GPU handoff. Revisit one family only when baseline profiling identifies it as the bottleneck and connected-caller/CPU-ISA/CUDA/HIP parity checks can run. This bootstrap does not rewrite kernels to reproduce older benchmark numbers. |
| Dense MMQ and input fix: `8eb2e1976982a224589e1cd215569a3079e188fa`, `b33220a8b2d013824bb19f64da179520c80a404b` | **Capability evolved; detailed port deferred.** The baseline uses `STRATA_DENSE_MMQ`/`Gemm::native` and K-quant instances rather than this fork's `STRATA_PREFILL_DENSE_MMQ` plumbing. Revisit only with a failing dense-projection parity case or measured gap, tracing the attention gate and hyper-connection inputs before adapting any fix. |
| Helper caches, layer split, adaptive/pinned RAM tier, tensor-parallel prototype: `19d3bf5592379c542bf727fd6d96f5f77d465fc5`, `bba0d25abe1d2501f38031f44de94912c34bb687`, `911bdeb64641e6f5b330117b4e551b3e556f89e1`, `390dd9d0f56fa5460cff2c296910135b6e8d6cdc`, `2c4f312aa2034b472dc45b4f4e21282457bc3895`, `5a179cac43766f5bab6e7894921fc6d33c1c874d`, `3cf89dd3251f72ca041799797c244714bd114694`, `ecfc0a9f90a75db1fc2f056b1de0fb7a7d5c0917`, `e3edf93d02deb37d1acdb9fda5779599c54004de`, `558aecbea9f7e45dbcc4a24cb2e179adc657868a` | **Deferred / outside first qualification.** Additional devices change peer access, memory ownership, streams, pinned-pool sizing and model state. Start with a single R9700. Revisit only for an explicitly approved homogeneous multi-GPU target after that baseline, preserving existing capabilities rather than enabling a new helper policy globally. Mixed-vendor execution is not authorized. |
| Long-session changes: `957dbf4aaf524322e726fb02560b4fe02f1eeff4` | **Split disposition.** Python server sampling/thinking continuation is **out of scope**, not an AMD kernel requirement. Helper-buffer growth fix and POSIX stack-reporting diagnostics are **deferred**: the former depends on the helper-compute code above, the latter changes signal/thread behavior. Revisit with a concrete equivalent failure in the newer baseline; do not import a broad server behavior change to obtain a buffer fix. |
| Broad upstream merge: `f1c4d045488c11e539f69a23a8101669950c4c07` | **No merge; full internal review deferred.** It brings Strata 0.1.30, conversation-cache work, Docker/setup/server/UI changes and earlier RDNA4 integration into the fork. It is not a cohesive AMD optimization. The newer UnbiasedStrata snapshot is authoritative here. Revisit individual missing fixes only after reconciling each merged source against that baseline; no full-review bookmark is claimed for this merge. |
| Benchmark records and deployment prose: `5578af5de98e9d479033b6d13dc7b727e9314f0c`, `ed3612f7c6e74ae8af31b64fefced1c4260e2ab8`, `3a996f2c5cbfd029dbfcd2fac7b473bbe3fee9e6`, `e128da05bb493796a51cab0c30aa331ba6648db1` | **Reference only; out of package.** They document the fork's own two-R9700/UD-Q4_K_XL setup and attribution. They do not qualify Redrock or justify importing its setup/download/UI machinery. Preserve the source links and credit, not private run artifacts or deployment assumptions. |

### Qualification and retained boundaries

The [1.0.0 record](releases/1.0.0.md) supersedes the earlier CPU-only checkpoint: the exact engine/adapter pair and repaired ROCr passed attended synthetic parity and IQ2_XS/128K MTP OFF/ON functional checks, with independent review PASS and explicit accepted limitations. This result records kernel 7.3.3 with mmap experts on the selected R9700; it is historical qualification evidence, not a host kernel pin. New work targets and validates the running platform. RX 7700 XT and A3000 still need separate compilation and device/model evidence.

The qualified ROCr adaptation is based on `ROCm/ROCR-Runtime` `4e4d7abeb786e903ac2a46815ae2565de331cbd4` (7.1.1), repaired as `f999715d0b28afdb4e9c3eb31317ff7789be54c9`. The [retained patch and reconstruction recipe](releases/1.0.0.md#repaired-rocr) preserve the source independently of that local commit. Stock ROCr failed selected-device initialization on this host; exposing additional GPUs is not an equivalent fix. This is a platform dependency repair, not an imported strata-rdna4 kernel change.

The supported wrapper still packages only the engine, with static ggml and a matching CUDA K-quant sidecar where applicable. The qualified headless adapter is the retained same-tree `unbiased-strata` target over the token protocol (`frontend/engine.hpp`, `docs/BACKENDS.md`), built separately. Release assembly must preserve that pair and its repaired runtime closure. No self-contained redistributable or managed installation/serving integration is established by the candidate tests.

## Repeatable update procedure

“Check whether there are updates worth incorporating since last time” invokes **check and recommend** below. It does not authorize incorporation, GPU execution, host updates or publication. The two engine feeds are **robertkarl/UnbiasedStrata** and **vectorweft/strata-rdna4**. Inspect llama.cpp changes proposed by the engine feed as dependency changes, not an independent instruction to follow latest llama.cpp. Check the ROCr repair against upstream when relevant to a runtime change or possible removal of the local patch.

### Checkpoints

[redrock.lock.json](redrock.lock.json) is the machine-readable checkpoint. For UnbiasedStrata, `commit` is the imported snapshot, `imported_commits` lists subsequent selective imports, and `reviewed_through` is the fully assessed boundary. For strata-rdna4, `commit` is only the frozen comparison head; `imported_commits` names actual imports and `reviewed_through` is intentionally older. ROCr records its source base, local repair and patch; it is required by the declared R9700 configuration, not automatically built by the engine wrapper.

Keep three facts distinct: **observed head** (seen), **reviewed through** (every change assessed) and **incorporated source** (actually shipped). Until a source has a `last_observed` entry, its frozen `commit` is the last observed head. After a completed new check, append a dated entry here with previous/new observed hashes, assessed ranges, dispositions and proposed work; update `last_observed` in the lock as `{"date": "YYYY-MM-DD", "commit": "<full hash>"}`. A no-change check also records that result. Keep imported pins unchanged until incorporation. A review can defer code while advancing its bookmark only if every change in the range has been assessed and the deferred ledger remains explicit. Existing deferred work is not silently cleared by checking newer commits.

### Check and recommend

1. Read the checkpoints, latest release and retained deferred families. Inspect the maintained tree and any uncommitted work. **Done:** the current shipped inputs, review debt and requested scope are known.
2. Resolve both tracked engine refs to immutable heads and fetch those objects using the host's authorized remote-access helper. Compare each source to its own history, verifying ancestry with `git merge-base --is-ancestor`. Missing/rewritten history stops that source's comparison for human reconciliation; no guessed common base or whole-fork merge. **Done:** old/new observed heads and bounded ranges are recorded, or a concrete access/history blocker is reported.
3. Assess the new range since the last observed head and any unreviewed gap from `reviewed_through`; report the gap separately if not fully assessed. Group cohesive families, but read actual changes, connected callers, tests and merge contents, not just subjects. Classify each as **incorporate**, **already present/superseded**, **defer** (with revisit condition), or **reject/out of scope**. Prioritize correctness/security, then measured benefit on the supported configuration; preserve architecture guards and newer baseline behavior. **Done:** recommendations name source hashes, rationale, affected contracts and required checks; no claim of full review crosses an unassessed gap.
4. Record the check and valid checkpoint advances in this file and the lock on a feature branch, leaving source/dependency pins untouched. A partially assessed check records its exact limits. Return a concise recommendation, including “nothing worth incorporating” where appropriate. **Done:** the next agent can resume without repeating completed assessment or forgetting deferred work.

### Incorporate an approved update

1. Start a feature branch from the maintained branch and apply only justified cohesive work. Use `git cherry-pick -x` where appropriate; manual adaptations name canonical repositories, full source hashes and changed assumptions. Record each imported upstream hash, downstream commit and rationale here; append selective engine imports to that source's `imported_commits` without advancing the whole-snapshot pin. Advance the imported snapshot only for a complete snapshot import; update any actual dependency pin and gitlink together. Dependency patches are prepared as reviewed source before configuration, never substituted at configure time. **Done:** the maintained delta and source provenance are reviewable; no unassessed head is implied to be imported.
2. Verify source cleanliness/acquisition boundaries, selected backend compilation, connected correctness checks and affected qualified hardware/model cases. Preserve deferred items, failures and prior release evidence. Update this record, lock/gitlink and support claims together. Follow [RELEASING.md](RELEASING.md) only when preparing a new release. **Done:** a reproducible candidate and its actual evidence or precise blockers are recorded; existing release tags and installations remain unchanged.

# Inherited UnbiasedStrata provenance

## Sources

| What | Where | Version |
| --- | --- | --- |
| Hy4-Flash text runner, S-quant codecs and native container reader | [Niko1221/Strata](https://github.com/Niko1221/Strata/tree/6e38f588a646c0ee6948a74afbe670eb75896106/engine/src/hy4flash) | `6e38f588a646c0ee6948a74afbe670eb75896106`. Build changes, file mapping and measurements are in [docs/HY4.md](docs/HY4.md). |
| Engines (`strata`, `strata-poly`), tools, docs | https://github.com/VecSzn/PolyStrata | tag `v0.1.45-poly.1`, commit `d0cd84dc6bd009d1d9e2191e4e28cc76b80f1ee3` |
| ggml, llama, common, mtmd, cpp-httplib | https://github.com/robertkarl/llama.cpp, branch `UnbiasedStrata`, as the submodule `third_party/llama.cpp` | commit `385445fad49e42fa718b5e1f71668f35b7a08559`: ggml-org `3cf03257f219afbe7334045ff7c6a06ac68c627d` (the commit upstream pins) plus the six parser commits below |
| DeepSeek-V4.1 engine (`strata-dsv41`: `src/dsv41`, `include/strata/dsv41`, `tests/dsv41`, `tools/dsv41`, `data/dsv41`, `tools/trellis/tqenc.*`), the trellis and `iq*x_rows` CPU kernels it needs | https://github.com/Niko1221/Strata | tag `v0.2.4`, commit `96ca4383` (2026-10-24); see "DeepSeek-V4.1 engine" below |
| llama.cpp's DeepSeek V4.1 chat template detection and DSML parser | https://github.com/ggml-org/llama.cpp pull 30817, head `b72050d0` | the `common/chat.cpp`, `common/chat.h`, `common/parsers/deepseek.cpp` hunks only, as commit `44072a1` on the submodule branch `deepseek41-parser` (on `385445f`) |
| Inkling 2 Mini CUDA engine | [Project Maya fork](https://github.com/alesha-pro/maya-inkling-mini/tree/e819852f822479af783638a467bfdfe2ed6882ec) | `src/ik2`, integrated with the GGUF shard reader and the front-end protocol |
| Inkling 2 Mini Metal engine | [Strata v0.2.19](https://github.com/Niko1221/Strata/tree/c6f6427d9de9016fd92d13e29740f65625261df7/engine/src/inkling2) | `third_party/strata-inkling`, the native Metal runner, SQ reader, codecs and dependencies. The fork accepts projected image rows from the front end and bounds multi-row logit requests to its output buffer. |
| Inkling 2 vision projector | [llama.cpp b12590](https://github.com/ggml-org/llama.cpp/tree/34ff2e047ce520b1d83bd274bf3ff0bfbc72a69b/tools/mtmd) | mtmd backport and typed image blocks on the llama.cpp submodule branch `strata-inkling2-vision`, commit `db1f382bb` |

Commit `0197ae86` ("Import PolyStrata v0.1.45-poly.1 (d0cd84dc) unmodified") is the upstream tree, every tracked
file, with the same tree hash as upstream's commit. `git diff 0197ae86 HEAD` is the whole story of this fork;
`git diff 0197ae86 HEAD -- src include` is every change to the engines.

The history before that commit is UnbiasedStrata 1, a fork of Niko1221/Strata 0.1.39, ended at the tags
`unbiasedstrata-1` and `unbiasedstrata-1-muse-spark-native`.

## Rules

A thing is removed only when it (1) reaches the network during setup, build or start-up, (2) puts Python
between a request and a token, (3) breaks or complicates the one-command build, or (4) is marketing.
Everything else stays, built or not. Engine changes are bug fixes for the target (Linux x86-64, one NVIDIA
GPU, Pascal or newer, CUDA 12.x), each recorded below with the file, the reason and the measurement.

## Build (`CMakeLists.txt`, `Makefile`, `.gitmodules`, `.gitignore`)

- **ggml from the submodule, not a fetch.** `STRATA_GGML_DIR` defaults to `third_party/llama.cpp` and the
  configure stops with the `git submodule update --init` line if `ggml/CMakeLists.txt` is missing. Upstream's
  `FetchContent` of llama.cpp (the only network access of its build) is deleted. The three later
  `if(STRATA_GGML_DIR)` blocks are unchanged; their `else()` branches are now dead. `.gitignore` no longer
  ignores `/third_party/llama.cpp*`.
- **llama.cpp changes are commits on a fork, not configure-time patches.** The submodule points at
  robertkarl/llama.cpp, branch `UnbiasedStrata`: ggml-org `3cf03257` plus the two commits under "Changes to llama.cpp"
  below. Until 2026-10-19 the same changes were `patches/*.patch`, applied to the submodule's working tree by CMake at
  configure (UnbiasedStrata 1 did it in `make` and in CMake both), which left the submodule dirty after every build and
  meant the pinned commit alone did not name the sources compiled. The pin now does; `git status` is clean after a build.
- **llama.cpp's libraries for the front end.** A second `add_subdirectory` of the submodule with
  `LLAMA_BUILD_COMMON` and `LLAMA_BUILD_MTMD` on and tests, tools, examples, server, app, UI, prebuilt UI,
  OpenSSL and llguidance off. It reuses the ggml targets configured above. Gated by `STRATA_FRONTEND` (default on),
  not by CUDA, so the front end also builds on a machine without a GPU.
- **The front end, `unbiased-strata`.** Seven of llama.cpp's `tools/server` files are compiled unmodified from the
  submodule into a static library (`server-http`, `server-queue`, `server-task`, `server-common`, `server-chat`,
  `server-schema`, `server-stream`): the HTTP layer, the task queue, the OpenAI request and result types, SSE
  framing, the request schema. The eighth, `server-context.cpp`, is bound to `llama_context` and is replaced by
  `frontend/strata_context.cpp`, which runs the engine as a child process over its line protocol (`docs/BACKENDS.md`)
  and uses a vocab-only `llama_model` for tokenizing, the chat template and the output parsers. `server-http.cpp`
  references the web UI's asset table (`ui.h`, generated by `tools/ui` at build time, which downloads); `frontend/
  ui_stub/` provides an empty one. No patch to llama.cpp was needed for this; the undocumented internal API of
  `tools/server` is used as it is at the pinned commit.
- **`Makefile`.** `make build` configures once and builds `strata`, `strata-poly`, `strata_kq` and `unbiased-strata`
  into `build/bin/`, not upstream's `all`, which also builds about 70 parity and bench programs. `make frontend`
  builds the front end alone without CUDA into `build-frontend/`; `make test` builds and runs the front end's tests
  there against a fake engine (`frontend/tests/`): the adapter's cases and the HTTP surface a coding agent touches.
  The nvcc check applies only to the `build` and `configure` goals.
  `CUDA_ARCHS` defaults to `61;75;86;89`; an arch below 75 adds upstream's `STRATA_EXPERIMENTAL_SM60`
  (upstream's default is `120`, which needs CUDA 13). `PORTABLE=1` gives the AVX2 baseline instead of
  `-march=native`. One GCC for the C, C++ and nvcc host sides: the Makefile probes `g++-15`, `g++-14`, ... with
  `nvcc -ccbin` and passes the newest accepted one as `CMAKE_C_COMPILER`, `CMAKE_CXX_COMPILER` and
  `CMAKE_CUDA_HOST_COMPILER`. Without that, on a box whose default `g++` is newer than nvcc accepts (Debian 13 with
  CUDA 12.4: GCC 14 against a GCC 13 limit), the C++ files were compiled by GCC 14 and the CUDA files by GCC 13, and
  `strata` failed to link on `__cxa_call_terminate` (upstream's installer handles the same case, its #1645).
  `HOSTCXX=/usr/bin/g++-13` overrides the probe.

## Front end behaviour (`frontend/`), where it differs from llama-server or from upstream's server

Recorded here because a client written for one of the two may notice.

- **`/health`** returns llama.cpp's `{"status": "ok", "busy": ...}` plus upstream's `"service": "strata"`,
  `"loaded": true`, `"model"`, `"max_context"` (commit 453d3015). Upstream's `tools/strata_bench.py` refuses a server
  without the first two; the first step-9 fork arm was lost to that
  (the step-9 record's README; records are in history at 7551f138, `bench/unbiased/`).
- **Streams always end with a usage chunk** (empty `choices`, `usage`, `timings`), as upstream's server does,
  whether or not the client sent `stream_options.include_usage`; llama-server sends it only on request (96f4ef1f).
  Same reason: the bench client reads `prompt_tokens` from it.
- **One slot.** Requests are served one at a time in arrival order, as upstream's server does with one engine.
- **No constrained decoding.** `response_format` and tool schemas are parsed and used for output parsing, not
  enforced at the sampler, because the engine samples; logged once per such request.
- **Pictures are encoded in the front end** (`--mmproj FILE`, commit 71886db0 and after): llama.cpp's mtmd runs the
  projector and the front end writes upstream's `SVE1` record file and sends `GENI … image_pad=<id> <file>`, the
  protocol upstream's Python server spoke to `strata-vision`. Projector width is checked against the model's
  embedding width at start. The default row budget is upstream's installer's (300 on the CPU, 1024 on the GPU;
  `--image-max-tokens` overrides). The encoder runs on the CPU: the front end's ggml is built without CUDA, and
  `--mmproj-gpu` is refused when ggml has no GPU device (17e27213). Measured in
  2026-10-18 on the A6000 (record in history at 7551f138).
- **`--sse-ping-interval`** is llama-server's (default 30 s); the measured Flash-Next runs used 10.
- **SIGTERM exits 1**, not 0, after stopping the engine: the engine child dies with the process group and the front
  end reports it as an engine death with an empty message. Open; matters only to a service manager.
- **Default `min_p`** is llama-server's 0.05 when a request omits it; upstream's server sends none. pi sends 0.

## Changes to llama.cpp (robertkarl/llama.cpp, branch `UnbiasedStrata`)

Six commits on top of ggml-org `3cf03257`; `git -C third_party/llama.cpp diff 3cf03257 HEAD` is exactly these.

- `9547d8d` (`common/parsers/qwen3-coder.cpp`): the Qwen3-Coder
  tool-call grammar accepts a call's arguments in any order. Upstream takes the required ones first and the
  optional ones after; Qwen3.8 Flash-Next often writes the optional ones first (`limit`, `offset`, `path`), and
  because the grammar's tool-call group is optional and has no end anchor, such a call was dropped silently:
  the client got `finish_reason=stop` and an empty answer (UnbiasedStrata 1, 2026-10-08, REGINA-104). Qwen3.6's
  template routes to the same parser. Carried over from UnbiasedStrata 1 unchanged.
- `a6de9af` (`common/chat-auto-parser-generator.cpp`, `common/parsers/deepseek.cpp`): the same for every
  template the auto-generated PEG grammar serves (DeepSeek V4 Flash's DSML, GLM 5.3 Flash's `<arg_key>` form, and
  the rest). Upstream accepts the required arguments only in the schema's order, which holds when its own sampler
  is constrained by the grammar; this engine samples unconstrained, and both models wrote `content` before `path`
  for a `write` tool that declares `path` first, so a finished, well-formed call was reported to the client as
  "output does not match the expected peg-native format" (PLAN.md step 13, 2026-10-19, USTRAT-7). Whether every
  required argument is present is left to the tool, as upstream already leaves it for optional-only tools.

- `1421694` (`tools/server/server-task.cpp`): a partial re-parse that reports fewer tool calls than the previous
  one no longer throws (the request died with a 500 mid-stream, "Invalid diff: now finding less tool calls!"); the
  previous message is kept, nothing is emitted for that token, and the raw generated text is logged. A non-partial
  parse failure returns the raw text as content instead of failing the request (issue #17, 2026-10-20).
- `638a7e2` (`common/parsers/qwen3-coder.cpp`, `tests/test-chat.cpp`): the Qwen3-Coder grammar skips a stuttered
  `</parameter>` close tag. Qwen3.8 Flash-Next Coder IQ1_M, sampled without the grammar, wrote the tag twice before
  `</function>` in 7 of 673 requests of a BBW run (2026-10-20); each one failed the whole call and, through the diff
  check above, the request. The value still ends at the first close. Test in the Qwen3-Coder section of test-chat.
- `4174b21` (`common/parsers/qwen3-coder.cpp`, `tools/server/server-task.cpp`, `tests/test-chat.cpp`): the same
  shard also leaves a blank line between the last `</parameter>` and `</function>` (memcached-bisect-deep,
  2026-10-21); the grammar now skips blank lines between arguments and before `</function>` and `</tool_call>`.
  And the final, non-partial parse gets the same guard as the partial one: when it finds fewer tool calls than were
  streamed, the streamed calls stand (arguments as last parsed) and the raw text is logged, instead of the diff
  check throwing and the request ending in a 500 after the hold-delta guard had kept it alive. Two blank-line cases
  in test-chat.
- `385445f` (`common/parsers/qwen3-coder.cpp`, `tests/test-chat.cpp`): the third shape (prime-frontend, 2026-10-21):
  `</content>` and `</invoke>` after the real `</parameter>`, a blank line, `</parameter>` again. The stuttered-close
  rule becomes general: any close tag that is not `</function>` or `</tool_call>` is skipped between arguments.
## Engine source (`src/`, `include/`)

No changes: `git diff 0197ae86 HEAD -- src include` is empty (checked 2026-10-18). The engines measured in step 9 are
these sources built in this tree; their binaries differ from upstream's build of the same sources, and the step-9
record says so where it matters.

Step 12 (harness `bench/correctness/`; record in history at 7551f138, `bench/unbiased/correctness/`; 2026-10-18) checked these builds against llama.cpp `3cf03257`: upstream's
parity programs that cover the kernels in use (38 pass, `poly_expert_parity` fails its own bound on one UD-Q3_K_M expert),
and teacher-forced per-step distributions on 16 sequences per model (within llama.cpp's own CPU-vs-GPU spread on every set
and arm). Two upstream claims did not hold and are not repeated here: the draft block is not bitwise inert above 4-token
windows, and `strata`'s output is not reproducible across starts unless the expert cache is pinned. Tickets USTRAT-1 to 5.

One test-only hunk since (2026-10-19, correctness checks of DeepSeek V4 Flash and GLM 5.3 Flash): `src/poly/main.cpp`
`--tokens`, `--eos-ids` and `--follow` accept `@FILE` (the comma-separated ids read from that file), because a 31K-id
list is over Linux's 128 KiB single-argument limit. `tools/qwen35/q36_llama_dump.cpp` (upstream's llama.cpp-side parity
program, not part of the build) takes the same `@FILE` for `--tokens` and `--force`. Nothing on the serve path changes.

## DeepSeek-V4.1 engine (`src/dsv41`, from Strata v0.2.4)

Added 2026-10-24 to run DeepSeek-V4.1-Flash-1017 (ISTA-DASLab's GSQ-RCO GGUF sets) on the A6000 VM. Neither engine of
this fork reads a `deepseek41` GGUF (`strata-poly` knows `deepseek4`, V4 Flash), and the pinned llama.cpp predates the
architecture, so the engine is upstream Strata's third one, `strata-dsv41` - itself a port of Project Maya's `maya-ds4`
(MIT; `third_party/project-maya/NOTICE.md`, the vendored EXL3 / sampler / window kernels under `src/dsv41/vendor/`).

- **Taken unmodified from Niko1221/Strata `v0.2.4` (96ca4383):** `src/dsv41/`, `include/strata/dsv41/`, `tests/dsv41/`,
  `tools/dsv41/{pack_gguf,smoke,tune}.py`, `data/dsv41/` (the shipped hot-expert profiles), `tools/trellis/tqenc.{cpp,hpp}`
  (the tests' encoder), `third_party/project-maya/`, `third_party/ggml/ggml-common-tq.h`, and upstream's `docs/DEEPSEEK41.md`
  (rewritten for this fork afterwards). `diff -r` against the tag's tree is the check.
- **`src/kernels/cpu` and `include/strata/kernels/cpu`:** upstream's `iq_avx2.{cpp,hpp}`, `iq_avx2_rows.inl`,
  `iq_avx512.{cpp,hpp}` replace this fork's v0.1.45-poly.1 copies - they add the engine's CPU lane rows `iq256x_rows` /
  `iq512x_rows` (ggml's float order, so a token's dot is ggml-cpu's bit for bit) and, upstream states, leave `iq256_*` /
  `iq512_*` as they were (`iq_avx2_parity` is the check here) - and the trellis CPU lane `tq_cpu.{hpp,cpp}`,
  `tq_avx2.cpp`, `tq_avx512.cpp`, `tq_states.hpp`, `tq_computed.hpp` is new in `strata_kernels_cpu`. `expert_layout.hpp`
  stays at layout version 4: upstream's 5 only adds Hadamard-folded packs, which nothing here makes or reads.
  `iq_avx512.cpp` is built with `-mavx512vpopcntdq -mbmi2` as upstream does (`iq512x_rows` uses them).
- **`CMakeLists.txt`:** upstream's `strata-dsv41` block as it is, but ggml.h / ggml-cpu.h come through
  `strata_kernels_cpu`'s `ggml-cpu` link from the submodule (no FetchContent, rule 1), and the option is turned on by the
  `Makefile` (`STRATA_BUILD_DSV41=ON`; `strata-dsv41` is in `TARGETS`). The engine is compiled with
  `STRATA_DSV41_TQ_CPU=1`, the ggml-cpu lane the IST-DAS sets' experts beyond VRAM run on.
- **Not taken:** upstream's `setup.py` / `strata launch` plumbing (rules 1 and 2), the signed engine-package feed, the
  Python server's `serve/dsml.py` (this fork's front end parses DSML with llama.cpp's `common/parsers/deepseek.cpp`,
  below) and the fixture template `serve/fixtures/deepseek41/chat_template.jinja` (the GGUF's own template is used).
- **Serving:** the front end starts it like the other two, `--engine strata-dsv41 -m <shard 1> ... -- --pack <pack>
  --max-context N --dspark auto`; `--serve` is added. Its line protocol is the one `frontend/engine.hpp` speaks
  (`READY`, `INFO`, `T`, `DONE`, `ERR`); its extra `STAT` and `NOTE` lines are ignored as `OTHER`. The front end's
  vocabulary load falls back to the `general.architecture` override (USTRAT-6's path) because the pinned llama.cpp
  does not know `deepseek41`.

## Removed

Each entry names the rule above that it satisfies.

- **Rule 1, installers and downloaders:** `setup.py` (8,223 lines: hardware detection, pip, CUDA, llama.cpp zips,
  prebuilt engines from GitHub releases, model files from Hugging Face or ModelScope, pack and MTP orchestration,
  run-config JSON, start), `setup.sh`, `SETUP.bat`, `START-HERE.bat`, `UPDATE.bat`, `update.sh`, `Dockerfile`,
  `docker-entrypoint.sh`, `.dockerignore`, `flake.nix`, `flake.lock`, `requirements.txt` (replaced by
  `tools/requirements.txt`: numpy and regex, for the offline tools only), `conftest.py` (pytest root for
  `setup.py`'s tests), `tools/strata_mcp.py` (an installer driven over MCP), `tools/image/` (stable-diffusion.cpp,
  Vulkan SDK and model downloads), `tools/hip/package_windows.py` (AMD nightly wheels), and their tests
  (`tools/test_setup_*.py` with two golden JSONs, `test_strata_mcp.py`, `test_qimage.py`,
  `test_docker_entrypoint.py`, `test_poly_prebuilt.py`). The jobs of `setup.py` that were not downloads (choosing
  engine flags per card, RAM and context) become documented flags in `README.md`.
- **Rule 2, Python in the serving path:** `serve/` (92 files, 33.9k lines: the HTTP server with the OpenAI,
  Anthropic and Responses dialects, the jinja2 chat templating, the Python tokenizer path, the per-family
  reasoning and tool-call parsers, MCP, structured output, replicas, Prometheus, the hardware monitor, the web
  UI and 58 test files) and `polystrata.py` (the terminal front end, which imports `serve/`). Replaced by the C++
  front end in `frontend/`, built from the pinned llama.cpp's `common`, `mtmd`, `cpp-httplib` and the
  engine-agnostic files of `tools/server`. The engines' stdin/stdout protocol is unchanged.
- **Rule 4, marketing:** the six translated READMEs (383 KB), `docs/media/` (12 MB: a 9.6 MB demo video still,
  a 1.8 MB GIF, the figure SVGs and their generator), `.github/` (issue and PR templates), `SECURITY.md` (the
  upstream project's advisory process) and `AGENTS.md` (contributor notes for upstream's tests and installers).
  `README.md` is rewritten for this fork (PLAN.md step 10); the fork's model-file record is `docs/MODEL-FILES.md`,
  its measurements are the "Measured" section of `README.md` (records in history at 7551f138).
- **Documents only removed components (decided 2026-10-17):** `docs/INSTALL.md`, `AI_SETUP.md`, `MCP_SERVER.md`,
  `IMAGES.md`, `LLAMA_SWAP.md`, `COMMUNITY_BENCHMARKS.md`, `FAQ.md`; `chat.py` (a terminal client) and
  `tools/calibrate.py` (imports `setup` and `serve.server`), with their tests.
- **Scripts that imported removed modules (decided 2026-10-17):** the engine gate scripts that drove the engine
  through the Python server's `StrataEngine` class (`tools/batch_groups_cache_test.py`, `cancel_park_test.py`,
  `conversation_cache_*.py`, `conversation_restore_speed.py`, `logit_bias_english_example.py`,
  `logit_bias_http_probe.py`, `session_pressure_test.py`), `tools/strata_bench.py` with `bench.sh` and `BENCH.bat`
  (one helper from `setup`), `tools/make_poly_release.py`, and the tests that imported `setup` or `serve`
  (`test_mtp_fetch.py`, `test_shards.py`, `test_strata_tokenizer.py`, `test_effort_end_args.py`,
  `test_research_run.py`, `test_benchmark_prompt.py`, `test_dll_closure.py`, `test_windows_amd_gpu.py`, the
  `test_conversation_cache_*` and `test_strata_bench.py`). Upstream's copies run against upstream.
- **Per-machine results and logs (decided 2026-10-19):** upstream's `bench/results/` (23 MB, one directory per
  community machine) and the fork's own `bench/unbiased/` run directories (57 MB). A result from one machine is a
  record, not source; the numbers that matter are in `README.md` "Measured", and the directories are in history at
  7551f138. `bench/` keeps what is reusable: `bench_vs_llama.py` with `profile_corpus.json`, the kernel probes, and
  `bench/correctness/` (the step-12 harness, moved from `bench/unbiased/correctness/scripts/`).
