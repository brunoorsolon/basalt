#!/usr/bin/env python3
"""Explicit dependency preparation and fresh, engine-only Linux candidates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEPENDENCY = Path("third_party/llama.cpp")
TARGETS = {"hip": ("gfx1201", "gfx1101"), "cuda": ("86",)}


def run(args, cwd=ROOT, env=None):
    return subprocess.check_output([str(a) for a in args], cwd=cwd, env=env, text=True)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def pins():
    lock = json.loads((ROOT / "redrock.lock.json").read_text())
    entry = run(["git", "ls-tree", "HEAD", "--", DEPENDENCY]).split()
    require(entry[:2] == ["160000", "commit"] and entry[2] == lock["llama_cpp"]["commit"],
            "Dependency lock differs from HEAD's gitlink; review and commit the pins together.")
    return lock


def clean(repo):
    require(not run(["git", "status", "--porcelain", "--untracked-files=all", "--ignored"], repo).strip(),
            f"Source is not clean: {repo}. Commit changes and move generated files outside the checkout.")


def identity():
    lock = pins()
    require((ROOT / DEPENDENCY / ".git").exists(),
            "Dependency is missing; run python3 tools/redrock.py prepare explicitly (network required).")
    require(run(["git", "rev-parse", "HEAD"], ROOT / DEPENDENCY).strip() == lock["llama_cpp"]["commit"],
            "Dependency revision mismatch; preserve local work and check out the locked commit explicitly.")
    clean(ROOT / DEPENDENCY)
    clean(ROOT)
    return {"redrock_commit": run(["git", "rev-parse", "HEAD"]).strip(), "sources": lock}


def prepare():
    pins()
    if not (ROOT / DEPENDENCY / ".git").exists():
        require(not (ROOT / DEPENDENCY).exists() or not any((ROOT / DEPENDENCY).iterdir()),
                "Dependency directory is nonempty; preserve it before initializing the pinned submodule.")
        subprocess.run(["git", "submodule", "update", "--init", "--checkout", "--", str(DEPENDENCY)],
                       cwd=ROOT, check=True)
    identity()
    print("Pinned dependency is ready. Build does not acquire dependencies.")


def executable(value):
    path = shutil.which(value)
    require(path, f"Required tool '{value}' is missing. Provision it explicitly; Redrock installs nothing.")
    return str(Path(path).resolve())


def build_environment():
    overrides = sorted(name for name, value in os.environ.items() if value and (
        name.startswith(("CMAKE_", "NVCC_")) or name in {
            "CC", "CXX", "CUDACXX", "CUDAHOSTCXX", "HIPCXX", "HIPHOSTCXX",
            "CFLAGS", "CXXFLAGS", "CUDAFLAGS", "HIPFLAGS", "CPPFLAGS", "LDFLAGS"}))
    require(not overrides,
            f"Unset ambient build overrides: {', '.join(overrides)}. Select compilers with --cc, --cxx and --gpu-compiler.")
    # Preserve only platform discovery/loading inputs; never pass compiler or build-system injection variables.
    return {name: value for name, value in os.environ.items() if name in {
        "PATH", "HOME", "TMPDIR", "LD_LIBRARY_PATH", "CUDA_PATH", "ROCM_PATH", "HIP_PATH"}} | {"LC_ALL": "C"}


def configure_args(args, build, compilers):
    flags = {
        "CMAKE_BUILD_TYPE": "Release", "CMAKE_C_COMPILER": compilers["cc"],
        "CMAKE_CXX_COMPILER": compilers["cxx"],
        "CMAKE_RUNTIME_OUTPUT_DIRECTORY": str(build / "bin"),
        "CMAKE_LIBRARY_OUTPUT_DIRECTORY": str(build / "bin"),
        "CMAKE_SKIP_RPATH": "ON", "STRATA_GGML_DIR": str(ROOT / DEPENDENCY),
        "STRATA_PORTABLE": "ON", "STRATA_NATIVE_EXPERTS": "ON",
        "STRATA_FRONTEND": "OFF", "STRATA_BUILD_TESTS": "OFF",
        "STRATA_BUILD_DSV41": "OFF", "STRATA_BUILD_IK2": "ON" if args.engine == "strata-ik2" else "OFF",
        "STRATA_ENABLE_CUDA": "ON" if args.backend == "cuda" else "OFF",
        "STRATA_ENABLE_HIP": "ON" if args.backend == "hip" else "OFF",
        "STRATA_PREFILL_MMQ": "ON" if args.backend == "hip" else "OFF",
        "STRATA_MMQ_KQUANTS": "ON", "STRATA_Q6K_EXPERTS": "ON",
        "GGML_CPU_KLEIDIAI": "OFF", "GGML_OPENMP_FETCH": "OFF",
        "GGML_CCACHE": "OFF", "FETCHCONTENT_FULLY_DISCONNECTED": "ON",
        "FETCHCONTENT_UPDATES_DISCONNECTED": "ON",
    }
    if args.backend == "cuda":
        flags.update(CMAKE_CUDA_ARCHITECTURES=args.arch, CMAKE_CUDA_COMPILER=compilers["gpu"],
                     CMAKE_CUDA_HOST_COMPILER=compilers["cxx"])
    else:
        flags.update(CMAKE_HIP_ARCHITECTURES=args.arch, CMAKE_HIP_COMPILER=compilers["gpu"])
    return [f"-D{k}={v}" for k, v in flags.items()]


def licenses(destination):
    for repo, relative in ((ROOT, Path()), (ROOT / DEPENDENCY, DEPENDENCY)):
        for name in run(["git", "ls-files", "-z"], repo).split("\0"):
            if name and Path(name).name.upper().startswith(("LICENSE", "NOTICE", "COPYING", "AUTHORS")):
                target = destination / relative / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(repo / name, target)


def build_candidate(args):
    require(platform.system() == "Linux" and platform.machine() == "x86_64",
            "This bootstrap supports Linux x86-64 build recipes only.")
    require(args.arch in TARGETS[args.backend],
            f"Architecture {args.arch!r} is not a Redrock {args.backend} target; choose {TARGETS[args.backend]}.")
    require(args.engine != "strata-ik2" or (args.backend, args.arch) == ("hip", "gfx1201"),
            "strata-ik2 candidates currently require HIP and gfx1201 (one R9700).")
    require(args.jobs > 0, "--jobs must be positive.")
    candidate = args.candidate.expanduser().resolve()
    require(not candidate.is_relative_to(ROOT) and not ROOT.is_relative_to(candidate),
            "Candidate must be outside the source checkout and must not contain it.")
    require(not candidate.exists(), "Candidate already exists; choose a new path. Existing installations are never replaced.")
    environment = build_environment()
    provenance = identity()
    cmake = executable("cmake")
    compilers = {"cc": executable(args.cc), "cxx": executable(args.cxx),
                 "gpu": executable(args.gpu_compiler or ("nvcc" if args.backend == "cuda" else "clang++"))}
    # A fresh directory prevents inherited CMake caches from enabling acquisition or another backend.
    candidate.mkdir()
    incomplete = candidate / "INCOMPLETE"
    incomplete.write_text("Not an installation: configure/build/package must complete successfully.\n")
    build = candidate / "build"
    flags = configure_args(args, build, compilers)
    targets = [args.engine] + (["strata_kq"] if args.backend == "cuda" else [])
    try:
        subprocess.run([cmake, "-S", str(ROOT), "-B", str(build), *flags], env=environment, check=True)
        require(identity() == provenance, "Source identity changed during configuration.")
        subprocess.run([cmake, "--build", str(build), "--parallel", str(args.jobs), "--target", *targets],
                       env=environment, check=True)
        require(identity() == provenance, "Source identity changed during compilation.")
        package = candidate / "runtime"
        binary = package / "bin"
        binary.mkdir(parents=True)
        outputs = [args.engine] + (["libstrata_kq.so"] if args.backend == "cuda" else [])
        for name in outputs:
            shutil.copy2(build / "bin" / name, binary / name)
        metadata = package / "share" / "redrock"
        metadata.mkdir(parents=True)
        licenses(metadata / "licenses")
        documents = ["README.md", "UPSTREAM.md"]
        if args.engine == "strata-ik2":
            documents += ["docs/INKLING_MINI_HIP.md", "docs/KERNEL_POLICY.md",
                          "docs/qualification/inkling-mini-hip-inputs.json",
                          "docs/qualification/inkling-mini-hip-qualification.json"]
            documents += [str(path.relative_to(ROOT)) for path in sorted((ROOT / "docs/qualification/inkling-mini-hip").iterdir())]
        for name in documents:
            target = metadata / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        require(identity() == provenance, "Source identity changed during packaging.")
        record = dict(provenance, backend=args.backend, architecture=args.arch, engine=args.engine,
                          cmake_version=run([cmake, "--version"], env=environment).splitlines()[0],
                          compiler_versions={k: run([v, "--version"], env=environment) for k, v in compilers.items()},
                          cmake_flags=flags, build_environment=environment, hardware_qualification="unqualified",
                          runtime_prerequisites="Linux/glibc, libstdc++, and the matching CUDA or ROCm toolkit and driver; see README.md")
        record["sha256"] = {name: hashlib.sha256((binary / name).read_bytes()).hexdigest() for name in outputs}
        (metadata / "build.json").write_text(json.dumps(record, indent=2) + "\n")
        require(identity() == provenance, "Source identity changed before candidate completion.")
        incomplete.unlink()
    except BaseException:
        print(f"Candidate is incomplete and retained at {candidate}; choose a new path for a retry.", file=sys.stderr)
        raise
    print(f"Candidate runtime: {candidate / 'runtime'} (unqualified; no activation performed)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="explicitly acquire the pinned submodule; never updates an existing checkout")
    commands.add_parser("check", help="verify committed dependency identities and clean source trees offline")
    build = commands.add_parser("build", help="configure, compile and package an engine in a new candidate directory")
    build.add_argument("--backend", choices=TARGETS, required=True)
    build.add_argument("--arch", required=True)
    build.add_argument("--engine", choices=("strata", "strata-poly", "strata-ik2"), required=True)
    build.add_argument("--candidate", type=Path, required=True)
    build.add_argument("--jobs", type=int, default=4)
    build.add_argument("--cc", default="cc")
    build.add_argument("--cxx", default="c++")
    build.add_argument("--gpu-compiler", help="nvcc for CUDA or ROCm clang++ for HIP (not hipcc)")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            prepare()
        elif args.command == "check":
            print(json.dumps(identity(), indent=2))
        else:
            build_candidate(args)
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"redrock: {error}\n")


if __name__ == "__main__":
    main()
