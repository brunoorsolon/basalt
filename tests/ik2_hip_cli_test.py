"""HIP admission refusals must happen before opening a model or initializing a GPU."""
import os
import subprocess
import sys

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
print("HIP Mini CLI admission refusals passed (no GPU/model required)")
