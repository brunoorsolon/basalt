#!/usr/bin/env python3
"""Explicit, attended, offline Inkling Mini HTTP probe. Never downloads or activates anything."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import re
import socket
import subprocess
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_inputs(candidate, adapter, model, manifest):
    require(not (candidate / "INCOMPLETE").exists(), "candidate is incomplete")
    record = json.loads((candidate / "runtime/share/redrock/build.json").read_text())
    require((record["engine"], record["backend"], record["architecture"]) ==
            ("strata-ik2", "hip", "gfx1201"), "not an R9700 Inkling candidate")
    engine = candidate / "runtime/bin/strata-ik2"
    require(sha256(engine) == record["sha256"]["strata-ik2"], "engine differs from build record")
    require(model.stat().st_size == manifest["model"]["size"], "model size differs from selected input")
    require(sha256(model) == manifest["model"]["sha256"], "model SHA-256 differs from selected input")
    return engine, {"build": record, "inputs": manifest, "adapter_sha256": sha256(adapter),
                    "model_sha256": manifest["model"]["sha256"]}


def request(url, body=None, timeout=120):
    data = None if body is None else json.dumps(body).encode()
    return urlopen(Request(url, data, {"Content-Type": "application/json"}), timeout=timeout)


def completion(url, body, streamed=False):
    deadline = time.monotonic() + 120
    with request(url + "/v1/chat/completions", {**body, "stream": streamed}) as response:
        if not streamed:
            result = json.load(response)
            return result["choices"][0]["message"], result
        text, calls, events, finished = [], {}, [], False
        for line in response:
            require(time.monotonic() < deadline, "completion exceeded its 120-second wall-clock limit")
            if not line.startswith(b"data: "):
                continue
            if line.strip() == b"data: [DONE]":
                finished = True
                break
            event = json.loads(line[6:])
            require("error" not in event, f"stream error: {event}")
            events.append(event)
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                text.append(delta.get("content") or "")
                for call in delta.get("tool_calls", []):
                    target = calls.setdefault(call["index"], {"id": "", "type": "function",
                                                             "function": {"name": "", "arguments": ""}})
                    target["id"] += call.get("id", "")
                    for key in ("name", "arguments"):
                        target["function"][key] += call.get("function", {}).get(key, "")
        require(finished, "stream ended without [DONE]")
        message = {"role": "assistant", "content": "".join(text)}
        if calls:
            message["tool_calls"] = [calls[i] for i in sorted(calls)]
        return message, events


class MemorySampler:
    """50ms process-tree RSS and device-wide VRAM sampling, not allocator-level peaks."""
    def __init__(self, pid, device):
        self.pid, self.device = pid, device
        self.stop = threading.Event()
        self.error = None
        self.peak_rss = self.peak_vram = 0
        self.baseline_vram = int((device / "mem_info_vram_used").read_text())
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()

    def sample(self):
        try:
            while not self.stop.is_set():
                pending, seen, rss = [self.pid], set(), 0
                while pending:
                    pid = pending.pop()
                    if pid in seen:
                        continue
                    seen.add(pid)
                    proc = Path(f"/proc/{pid}")
                    try:
                        status = (proc / "status").read_text().splitlines()
                        rss += sum(int(s.split()[1]) * 1024 for s in status if s.startswith("VmRSS:"))
                        for children in (proc / "task").glob("*/children"):
                            pending.extend(int(p) for p in children.read_text().split())
                    except FileNotFoundError:
                        pass
                self.peak_rss = max(self.peak_rss, rss)
                self.peak_vram = max(self.peak_vram, int((self.device / "mem_info_vram_used").read_text()))
                self.stop.wait(.05)
        except Exception as error:
            self.error = str(error)

    def finish(self):
        self.stop.set()
        self.thread.join()
        require(self.error is None, f"memory sampling failed: {self.error}")
        return {"peak_process_tree_rss_bytes": self.peak_rss, "baseline_device_vram_bytes": self.baseline_vram,
                "peak_device_vram_bytes": self.peak_vram, "sample_interval_seconds": .05,
                "limits": "RSS can double-count shared mappings; VRAM includes other device users; sampled peaks are lower bounds"}


def rates(log):
    samples = []
    pattern = r"DONE produced=(\d+) prompt=(\d+) prompt_ms=([\d.]+) decode_ms=([\d.]+) finish=(\w+) drafts=\d+/\d+ reused=(\d+)"
    for match in re.finditer(pattern, log):
        produced, prompt, pp_ms, gen_ms, finish, reused = match.groups()
        pp_ms, gen_ms = float(pp_ms), float(gen_ms)
        samples.append({"prompt_tokens_processed": int(prompt) - int(reused), "generated_tokens": int(produced),
                        "prompt_ms": pp_ms, "decode_ms": gen_ms, "finish": finish,
                        "prompt_tokens_per_second": (int(prompt) - int(reused)) * 1000 / pp_ms if pp_ms else None,
                        "generation_tokens_per_second": int(produced) * 1000 / gen_ms if gen_ms else None})
    return samples


def stop_server(proc):
    # This process group belongs only to this probe, including the adapter's engine child.
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)


def probe(args):
    manifest = json.loads(args.manifest.read_text())
    engine, evidence = verify_inputs(args.candidate, args.adapter, args.model, manifest)
    device = Path("/sys/bus/pci/devices") / args.pci
    require(device.is_dir() and (device / "vendor").read_text().strip() == "0x1002", "not an AMD PCI device")
    require((device / "device").read_text().strip() == "0x7551", "not the selected R9700 PCI device")
    require(args.device.startswith("GPU-"), "use an immutable ROCr GPU UUID, not an ordinal")
    require(not args.output.exists(), "output path exists; preserve earlier evidence and select a fresh path")
    args.output.mkdir(parents=True)
    evidence.update(status="incomplete", pci=args.pci, rocr_device=args.device,
                    kernel=os.uname().release, runtime_library_path=args.runtime_library_path,
                    configuration={"context": 8192, "chunk": 256, "kv": "f16", "drafting": False})
    env = {k: v for k, v in os.environ.items() if k in {"PATH", "HOME", "TMPDIR"}}
    env.update(ROCR_VISIBLE_DEVICES=args.device, HIP_VISIBLE_DEVICES="0", CUDA_VISIBLE_DEVICES="",
               LC_ALL="C", IK2_KV="f16", IK2_EXPERT_MMA="0", LD_LIBRARY_PATH=args.runtime_library_path)
    evidence["environment"] = env
    try:
        for iteration in range(2):
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            url = f"http://127.0.0.1:{port}"
            log_path = args.output / f"server-{iteration}.log"
            command = [str(args.adapter), "--engine", str(engine), "-m", str(args.model), "--jinja",
                       "--host", "127.0.0.1", "--port", str(port), "--parallel", "1", "--alias", "inkling-mini",
                       "--", "--model", str(args.model), "--max-context", "8192", "--chunk", "256"]
            with log_path.open("w") as log:
                start = time.monotonic()
                proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                memory = None
                try:
                    memory = MemorySampler(proc.pid, device)
                    deadline = start + 180
                    while True:
                        require(proc.poll() is None, f"adapter exited; inspect {log_path}")
                        require(time.monotonic() < deadline, "startup exceeded 180 seconds")
                        try:
                            with request(url + "/health", timeout=1) as response:
                                require(json.load(response).get("status") == "ok", "health is not ready")
                            break
                        except OSError:
                            time.sleep(.1)
                    require(f"hip device pci={args.pci.lower()} arch=gfx1201" in log_path.read_text().lower(),
                            "engine PCI device does not match the memory-sampling target")
                    evidence[f"startup_seconds_{iteration}"] = time.monotonic() - start
                    basic = {"model": "inkling-mini", "messages": [{"role": "user", "content": "Reply with exactly: ready"}],
                             "reasoning_effort": "none", "temperature": 0, "max_tokens": 32, "cache_prompt": False}
                    message, stream = completion(url, basic, streamed=True)
                    require(bool(message["content"].strip()), "streamed text was empty")
                    evidence[f"text_stream_{iteration}"] = stream
                    if iteration:
                        continue
                    tool = {"type": "function", "function": {"name": "lookup", "description": "Look up the given key",
                            "parameters": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}}}
                    messages = [{"role": "user", "content": "Call lookup with key redrock-check. Do not invent its result."}]
                    reply, tool_stream = completion(url, {**basic, "messages": messages, "tools": [tool], "max_tokens": 256}, True)
                    calls = reply.get("tool_calls", [])
                    require(len(calls) == 1 and calls[0]["function"]["name"] == "lookup", "missing/wrong tool call")
                    require(json.loads(calls[0]["function"]["arguments"]) == {"key": "redrock-check"}, "invalid tool arguments")
                    messages.extend([reply, {"role": "tool", "tool_call_id": calls[0]["id"], "content": "verification code: 739281"},
                                     {"role": "user", "content": "Reply with the verification code from the tool result."}])
                    followup, result = completion(url, {**basic, "messages": messages, "max_tokens": 64})
                    require("739281" in (followup.get("content") or ""), "tool-result continuation lost the result")
                    evidence.update(tool_stream=tool_stream, tool_result=result)
                    _, limited = completion(url, {**basic, "max_tokens": 8})
                    require(0 < limited["usage"]["completion_tokens"] <= 8, "output budget was not respected")
                    evidence["output_limit"] = limited
                    try:
                        completion(url, {**basic, "messages": [{"role": "user", "content": " x" * 9000}]})
                    except HTTPError as error:
                        require(error.code == 400, f"context refusal returned HTTP {error.code}")
                        evidence["context_refusal"] = {"status": error.code, "body": error.read().decode()}
                    else:
                        raise RuntimeError("oversized context was not refused")
                    before = log_path.read_text().count("finish=cancel")
                    with request(url + "/v1/chat/completions", {**basic, "stream": True, "max_tokens": 7000,
                                 "messages": [{"role": "user", "content": "Count from 1 to 5000, printing every integer."}]}) as response:
                        for line in response:
                            require(line.strip() != b"data: [DONE]", "generation ended before cancellation could be tested")
                            if line.startswith(b"data: {") and any(c.get("delta", {}).get("content") for c in json.loads(line[6:]).get("choices", [])):
                                break
                        else:
                            raise RuntimeError("no token to interrupt")
                    deadline = time.monotonic() + 20
                    while log_path.read_text().count("finish=cancel") <= before:
                        require(time.monotonic() < deadline, "engine did not acknowledge cancellation within 20 seconds")
                        time.sleep(.1)
                    recovered, recovery = completion(url, basic)
                    require(bool(recovered.get("content")), "request after cancellation failed")
                    evidence["after_cancellation"] = recovery
                finally:
                    stop_server(proc)
                    evidence[f"rates_{iteration}"] = rates(log_path.read_text())
                    if memory is not None:
                        evidence[f"memory_{iteration}"] = memory.finish()
        evidence["status"] = "functional_probe_passed_not_numerically_qualified"
        return evidence
    except BaseException as error:
        evidence["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (args.output / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "adapter", "model", "manifest", "output"):
        parser.add_argument("--" + name, type=lambda p: Path(p).expanduser().resolve(), required=True)
    parser.add_argument("--device", required=True, help="selected R9700 ROCr UUID (GPU-...)")
    parser.add_argument("--pci", required=True, help="selected PCI address (0000:03:00.0)")
    parser.add_argument("--runtime-library-path", required=True, help="explicit qualified ROCr/toolkit library path")
    parser.add_argument("--attended", action="store_true", required=True, help="operator explicitly authorizes this bounded real-model run")
    args = parser.parse_args()
    try:
        print(probe(args)["status"])
    except (RuntimeError, OSError, ValueError, KeyError) as error:
        parser.exit(1, f"qualification stopped: {error}\n")


if __name__ == "__main__":
    main()
