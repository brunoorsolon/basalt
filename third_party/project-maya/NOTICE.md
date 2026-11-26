# Project Maya (maya-ds4), DeepSeek-V4.1 engine

The files under `src/dsv41/`, `include/strata/dsv41/`, `tests/dsv41/` and `tools/dsv41/` come from
https://github.com/mw00/project-maya (release v1.1.17, commit 8d9a9b5d), its DeepSeek engine `src/ds4`, `tests/ds4_*`,
`tools/ds4` and the kernel sources it builds on (`exl3`, `sampler`). Project Maya is itself built on Strata.

Copyright (c) 2026 Peasant Smith (mw00) and the Project Maya contributors, including needmorevram.
MIT License, the full text is in `LICENSE` beside this file. The notice stays with every copy.

Changes made while porting into Strata are listed in the git history of the `src/dsv41` directory
(the first commit of each file is the unmodified upstream copy).

Later changes taken from Project Maya releases v1.1.18 to v1.1.26 (the same approach re-written onto Strata's engine,
for example the long-prompt layer skip, the thinking budget, conversation slots) are covered by the same copyright
and permission notice, in `LICENSE` beside this file.

## Inkling 2

The engine in `src/ik2/` and the kernel parity test in `tests/ik2_kernels_test.cu` are adapted from
[alesha-pro/maya-inkling-mini at e819852f](https://github.com/alesha-pro/maya-inkling-mini/tree/e819852f822479af783638a467bfdfe2ed6882ec).
The same Project Maya MIT license applies. Strata supplies the CMake target, GGUF shard resolution, and server integration.

Redrock's R9700 HIP candidate selectively adapts Project Maya at `53885cef8faf0d3b3f7e28c0b1537a056c1f3ecf` (https://github.com/mw00/project-maya): the portable CUDA-fragment and packed-half helpers from `src/ik2/ik2_kernels.cu`, prompt attention from `src/ik2/ik2_rdna.inc` into `src/ik2/ik2_rdna_attention.inc`, and `include/maya/kernels/rdna_wmma.hpp` into `src/ik2/ik2_rdna_wmma.hpp`. Namespace/build macros are adapted to Redrock. The same copyright and MIT permission notice apply. Maya's installer, Python server, model manager, offload lane and other engine updates are not imported. Source identity and qualification limits are recorded in `docs/INKLING_MINI_HIP.md` and `redrock.lock.json`.
