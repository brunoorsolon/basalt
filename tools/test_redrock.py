"""Build-orchestration regressions; fake toolchains never stand in for compilation evidence."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import redrock


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "source"
        self.root.mkdir()
        self.root_patch = patch.object(redrock, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        # run's default cwd is bound at definition time; route it to the fixture.
        original_run = redrock.run
        self.run_patch = patch.object(redrock, "run", lambda args, cwd=None, env=None: original_run(args, cwd or self.root, env))
        self.run_patch.start()
        self.addCleanup(self.run_patch.stop)
        self.git(self.root, "init", "-b", "fixture")
        dep = self.root / redrock.DEPENDENCY
        dep.mkdir(parents=True)
        self.git(dep, "init", "-b", "fixture")
        (dep / "LICENSE").write_text("Dependency license\n")
        self.commit(dep)
        self.pin = self.git(dep, "rev-parse", "HEAD").strip()
        (self.root / "redrock.lock.json").write_text(json.dumps({"llama_cpp": {"commit": self.pin}}))
        (self.root / "LICENSE").write_text("Engine license\n")
        (self.root / "UPSTREAM.md").write_text("Pinned upstream record\n")
        (self.root / "README.md").write_text("Runtime prerequisites\n")
        (self.root / ".gitmodules").write_text('[submodule "third_party/llama.cpp"]\n\tpath = third_party/llama.cpp\n\turl = https://example.invalid/no-download\n')
        self.commit(self.root)
        self.bin = self.base / "toolchain"
        self.bin.mkdir()
        tool = self.bin / "cmake"
        tool.write_text('''#!/usr/bin/env python3
import json, os, pathlib, sys
args = sys.argv[1:]
if args == ['--version']:
    print('Fixture toolchain, not a compiler')
elif '-S' in args:
    build = pathlib.Path(args[args.index('-B') + 1])
    build.mkdir(parents=True)
    (build / 'flags.json').write_text(json.dumps(args))
    (build / 'configure-environment.json').write_text(json.dumps(dict(os.environ)))
    source = pathlib.Path(args[args.index('-S') + 1])
    if (source.parent / 'mutate').exists():
        (source / 'LICENSE').write_text('unexpected configure mutation')
else:
    build = pathlib.Path(args[1])
    (build / 'build-environment.json').write_text(json.dumps(dict(os.environ)))
    binary = build / 'bin'
    binary.mkdir()
    for target in args[args.index('--target') + 1:]:
        name = 'libstrata_kq.so' if target == 'strata_kq' else target
        (binary / name).write_bytes(('fixture payload: ' + target).encode())
''')
        tool.chmod(0o755)
        self.path_patch = patch.dict(os.environ, {"PATH": str(self.bin) + os.pathsep + os.environ["PATH"]})
        self.path_patch.start()
        self.addCleanup(self.path_patch.stop)
        self.args = argparse.Namespace(backend="hip", arch="gfx1201", engine="strata", jobs=1,
                                      candidate=self.base / "candidate", cc="cmake", cxx="cmake", gpu_compiler="cmake")

    def git(self, repo, *args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL)

    def commit(self, repo):
        self.git(repo, "add", ".")
        self.git(repo, "commit", "-m", "Fixture source")

    def test_candidate_contains_only_selected_engine_metadata_and_licenses(self):
        redrock.build_candidate(self.args)
        runtime = self.args.candidate / "runtime"
        self.assertEqual((runtime / "bin/strata").read_bytes(), b"fixture payload: strata")
        self.assertEqual([p.name for p in (runtime / "bin").iterdir()], ["strata"])
        metadata = runtime / "share/redrock"
        record = json.loads((metadata / "build.json").read_text())
        self.assertEqual(record["redrock_commit"], self.git(self.root, "rev-parse", "HEAD").strip())
        self.assertEqual(record["sources"]["llama_cpp"]["commit"], self.pin)
        self.assertEqual(record["architecture"], "gfx1201")
        self.assertEqual(record["hardware_qualification"], "unqualified")
        self.assertEqual((metadata / "licenses/third_party/llama.cpp/LICENSE").read_text(), "Dependency license\n")
        self.assertFalse((self.args.candidate / "INCOMPLETE").exists())
        redrock.identity()

    def test_cuda_packages_its_matching_kquant_library_without_changing_target(self):
        self.args.backend, self.args.arch = "cuda", "86"
        self.args.engine = "strata-poly"
        redrock.build_candidate(self.args)
        binary = self.args.candidate / "runtime/bin"
        self.assertEqual({p.name for p in binary.iterdir()}, {"strata-poly", "libstrata_kq.so"})
        flags = json.loads((self.args.candidate / "build/flags.json").read_text())
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=86", flags)
        self.assertIn("-DSTRATA_ENABLE_HIP=OFF", flags)
        self.assertIn("-DFETCHCONTENT_FULLY_DISCONNECTED=ON", flags)
        self.assertIn("-DGGML_CPU_KLEIDIAI=OFF", flags)

    def test_inkling_candidate_selects_hip_engine_and_retains_maya_license(self):
        self.args.engine = "strata-ik2"
        notice = self.root / "third_party/project-maya/NOTICE.md"
        notice.parent.mkdir(parents=True)
        notice.write_text("Maya HIP slice provenance\n")
        (notice.parent / "LICENSE").write_text("Maya MIT license\n")
        guide = self.root / "docs/INKLING_MINI_HIP.md"
        guide.parent.mkdir()
        guide.write_text("Unqualified Inkling candidate reconstruction\n")
        inputs = self.root / "docs/qualification/inkling-mini-hip-inputs.json"
        inputs.parent.mkdir()
        inputs.write_text('{"status": "pending"}\n')
        (guide.parent / "KERNEL_POLICY.md").write_text("Target the running platform\n")
        qualification = inputs.parent / "inkling-mini-hip-qualification.json"
        qualification.write_text('{"status": "fixture historical proof"}\n')
        raw = inputs.parent / "inkling-mini-hip/api.json"
        raw.parent.mkdir()
        raw.write_text('{"fixture": true}\n')
        self.commit(self.root)
        redrock.build_candidate(self.args)
        runtime = self.args.candidate / "runtime"
        self.assertEqual({p.name for p in (runtime / "bin").iterdir()}, {"strata-ik2"})
        flags = json.loads((self.args.candidate / "build/flags.json").read_text())
        self.assertIn("-DSTRATA_BUILD_IK2=ON", flags)
        self.assertIn("-DSTRATA_ENABLE_CUDA=OFF", flags)
        self.assertIn("-DCMAKE_HIP_ARCHITECTURES=gfx1201", flags)
        self.assertIn("-DSTRATA_FRONTEND=OFF", flags)
        metadata = runtime / "share/redrock"
        self.assertEqual((metadata / "licenses/third_party/project-maya/NOTICE.md").read_text(), notice.read_text())
        self.assertEqual((metadata / "docs/INKLING_MINI_HIP.md").read_text(), guide.read_text())
        self.assertEqual((metadata / "docs/qualification/inkling-mini-hip-inputs.json").read_text(), inputs.read_text())
        self.assertEqual((metadata / "docs/qualification/inkling-mini-hip-qualification.json").read_text(), qualification.read_text())
        self.assertEqual((metadata / "docs/qualification/inkling-mini-hip/api.json").read_text(), raw.read_text())
        record = json.loads((metadata / "build.json").read_text())
        self.assertEqual(record["engine"], "strata-ik2")
        self.assertEqual(set(record["sha256"]), {"strata-ik2"})
        self.assertEqual(record["hardware_qualification"], "unqualified")

    def test_inkling_rejects_other_backend_or_architecture_before_creating_candidate(self):
        self.args.engine = "strata-ik2"
        for backend, arch in (("cuda", "86"), ("hip", "gfx1101")):
            with self.subTest(backend=backend, arch=arch):
                self.args.backend, self.args.arch = backend, arch
                with self.assertRaisesRegex(RuntimeError, "one R9700"):
                    redrock.build_candidate(self.args)
                self.assertFalse(self.args.candidate.exists())

    def test_build_environment_is_controlled_and_recorded(self):
        overrides = (
            "CMAKE_TOOLCHAIN_FILE", "CFLAGS", "CXXFLAGS", "CUDAFLAGS", "HIPFLAGS", "LDFLAGS", "CPPFLAGS",
            "CMAKE_C_COMPILER_LAUNCHER", "CMAKE_CXX_COMPILER_LAUNCHER", "CMAKE_CUDA_COMPILER_LAUNCHER",
            "CMAKE_HIP_COMPILER_LAUNCHER", "CMAKE_PREFIX_PATH", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS",
            "CC", "CXX", "CUDACXX", "CUDAHOSTCXX", "HIPCXX", "HIPHOSTCXX",
        )
        for name in overrides:
            with self.subTest(variable=name), patch.dict(os.environ, {name: f"override-payload-{name}"}):
                self.args.candidate = self.base / name
                with self.assertRaises(RuntimeError):
                    redrock.build_candidate(self.args)
                self.assertFalse(self.args.candidate.exists())
        self.args.candidate = self.base / "controlled-candidate"
        with patch.dict(os.environ, {"CPATH": "unreviewed-include-path", "BASH_ENV": "unreviewed-shell-script",
                                     "ROCM_PATH": "explicit-platform-path", "LC_ALL": "C"}):
            redrock.build_candidate(self.args)
        record = json.loads((self.args.candidate / "runtime/share/redrock/build.json").read_text())
        for stage in ("configure", "build"):
            consumed = json.loads((self.args.candidate / f"build/{stage}-environment.json").read_text())
            self.assertEqual(consumed, record["build_environment"])
            self.assertEqual(consumed["ROCM_PATH"], "explicit-platform-path")
            self.assertNotIn("CPATH", consumed)
            self.assertNotIn("BASH_ENV", consumed)
            self.assertTrue(set(overrides).isdisjoint(consumed))
        self.assertFalse((self.args.candidate / "INCOMPLETE").exists())

    def test_existing_installation_and_rollback_remain_untouched(self):
        self.args.candidate.mkdir()
        active = self.args.candidate / "model-or-rollback"
        active.write_bytes(b"keep me")
        with self.assertRaises(RuntimeError):
            redrock.build_candidate(self.args)
        self.assertEqual(active.read_bytes(), b"keep me")
        self.assertEqual(list(self.args.candidate.iterdir()), [active])

    def test_unsupported_architecture_does_not_fall_back_or_create_candidate(self):
        self.args.arch = "86"
        with self.assertRaises(RuntimeError):
            redrock.build_candidate(self.args)
        self.assertFalse(self.args.candidate.exists())

    def test_dirty_dependency_and_lock_mismatch_fail_before_configuration(self):
        (self.root / redrock.DEPENDENCY / "LICENSE").write_text("local work")
        with self.assertRaises(RuntimeError):
            redrock.build_candidate(self.args)
        self.assertFalse(self.args.candidate.exists())
        self.git(self.root / redrock.DEPENDENCY, "restore", "LICENSE")
        lock = self.root / "redrock.lock.json"
        lock.write_text(json.dumps({"llama_cpp": {"commit": "0" * 40}}))
        self.commit(self.root)
        with self.assertRaises(RuntimeError):
            redrock.build_candidate(self.args)
        self.assertFalse(self.args.candidate.exists())

    def test_missing_toolchain_never_acquires_or_creates_candidate(self):
        self.args.gpu_compiler = "redrock-absent-compiler"
        with self.assertRaises(RuntimeError):
            redrock.build_candidate(self.args)
        self.assertFalse(self.args.candidate.exists())

    def test_configure_source_mutation_leaves_only_incomplete_candidate(self):
        (self.root.parent / "mutate").touch()
        with self.assertRaises(RuntimeError), contextlib.redirect_stderr(io.StringIO()):
            redrock.build_candidate(self.args)
        self.assertTrue((self.args.candidate / "INCOMPLETE").exists())
        self.assertFalse((self.args.candidate / "runtime").exists())

    def test_prepare_of_ready_source_is_offline_and_does_not_move_dependency(self):
        with patch.object(redrock.subprocess, "run", wraps=subprocess.run) as calls:
            redrock.prepare()
        self.assertFalse(any(call.args[0][0:3] == ["git", "submodule", "update"] for call in calls.call_args_list))
        self.assertEqual(self.git(self.root / redrock.DEPENDENCY, "rev-parse", "HEAD").strip(), self.pin)


if __name__ == "__main__":
    unittest.main()
