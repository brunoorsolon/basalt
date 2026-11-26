"""HIP admission refusals must happen before opening a model or initializing a GPU."""
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

binary = sys.argv[1]
for options, reason in [
    (["--max-context", "0"], "--max-context"),
    (["--max-context", "8193"], "--max-context"),
    (["--chunk", "0"], "--chunk"),
    (["--chunk", "257"], "--chunk"),
    (["--gpus", "2"], "isolated R9700"),
    (["--draft", "/nonexistent/drafter"], "without drafting"),
    (["--pictures", "/nonexistent/image"], "text-only"),
    (["--serve-ik2"], "using --serve or --tokens"),
    (["--gen", "-1"], "--gen"),
]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("IK2_")}
    env["ROCR_VISIBLE_DEVICES"] = ""
    result = subprocess.run([binary, "--model", "/nonexistent/model", "--serve", *options],
                            env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode != 0, options
    assert reason in result.stderr, (options, result.stderr)
    assert "READY" not in result.stdout, options
for key in ("IK2_CPU", "IK2_KV_MANAGED", "IK2_TIER", "IK2_LAZY_CPU"):
    env = {k: v for k, v in os.environ.items() if not k.startswith("IK2_")}
    env.update(ROCR_VISIBLE_DEVICES="", **{key: "1"})
    result = subprocess.run([binary, "--model", "/nonexistent/model", "--serve"],
                            env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode != 0 and key in result.stderr, (key, result.stderr)

# A real, minimal GGUF from the wrong model family must fail before GPU discovery.
with tempfile.TemporaryDirectory() as directory:
    def string(value):
        encoded = value.encode()
        return struct.pack("<Q", len(encoded)) + encoded

    gguf = struct.pack("<IIQQ", 0x46554747, 3, 0, 1)
    gguf += string("general.architecture") + struct.pack("<I", 8) + string("llama")
    model = Path(directory) / "wrong-family.gguf"
    model.write_bytes(gguf + bytes((-len(gguf)) % 32))
    result = subprocess.run([binary, "--model", str(model), "--serve"],
                            env={"PATH": os.environ["PATH"], "ROCR_VISIBLE_DEVICES": ""},
                            capture_output=True, text=True, timeout=10)
    assert result.returncode != 0 and "requires inkling2 GGUF" in result.stderr, result.stderr
print("HIP Mini CLI admission refusals passed (14 cases, no GPU/weights required)")
