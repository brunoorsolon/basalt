"""Offline identity gates and streamed tool/result assembly; no hardware qualification claim."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import threading
import unittest

import qualify_inkling as q


class ProbeTests(unittest.TestCase):
    def test_inputs_are_exact_and_an_incomplete_candidate_is_never_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / "runtime/share/redrock/build.json"
            metadata.parent.mkdir(parents=True)
            engine = root / "runtime/bin/strata-ik2"
            engine.parent.mkdir(parents=True)
            engine.write_bytes(b"fixture engine")
            model, adapter = root / "model", root / "adapter"
            model.write_bytes(b"fixture model")
            adapter.write_bytes(b"fixture adapter")
            record = {"engine": "strata-ik2", "backend": "hip", "architecture": "gfx1201",
                      "sha256": {"strata-ik2": q.sha256(engine)}}
            metadata.write_text(json.dumps(record))
            manifest = {"model": {"size": model.stat().st_size, "sha256": q.sha256(model)}}
            self.assertEqual(q.verify_inputs(root, adapter, model, manifest)[0], engine)
            model.write_bytes(b"changed model")
            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                q.verify_inputs(root, adapter, model, manifest)
            model.write_bytes(b"fixture model")
            engine.write_bytes(b"changed engine")
            with self.assertRaisesRegex(RuntimeError, "engine differs"):
                q.verify_inputs(root, adapter, model, manifest)
            (root / "INCOMPLETE").touch()
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                q.verify_inputs(root, adapter, model, manifest)

    def test_probe_stops_only_its_own_process_group(self):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
        try:
            q.stop_server(proc)
            self.assertIsNotNone(proc.poll())
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    def test_rates_count_only_uncached_prompt_tokens(self):
        samples = q.rates("task 1: DONE produced=8 prompt=100 prompt_ms=100.0 decode_ms=200.0 finish=length drafts=0/0 reused=90")
        self.assertEqual(samples[0]["prompt_tokens_per_second"], 100)
        self.assertEqual(samples[0]["generation_tokens_per_second"], 40)

    def test_streamed_text_tool_fragments_and_result_continuation(self):
        bodies = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                bodies.append(body)
                self.send_response(200)
                self.end_headers()
                if body.get("stream"):
                    for delta in [
                        {"content": "Hello "}, {"content": "world"},
                        {"tool_calls": [{"index": 0, "id": "call-1", "function": {"name": "look", "arguments": '{"key":'}}]},
                        {"tool_calls": [{"index": 0, "function": {"name": "up", "arguments": '"x"}'}}]},
                    ]:
                        self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": delta}]}).encode() + b"\n\n")
                    if not body.get("truncate"):
                        self.wfile.write(b"data: [DONE]\n\n")
                else:
                    self.wfile.write(json.dumps({"choices": [{"message": {"role": "assistant", "content": "739281"}}]}).encode())

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}"
            message, _ = q.completion(url, {}, True)
            self.assertEqual(message["content"], "Hello world")
            call, = message["tool_calls"]
            self.assertEqual(call["function"]["name"], "lookup")
            self.assertEqual(json.loads(call["function"]["arguments"]), {"key": "x"})
            messages = [message, {"role": "tool", "tool_call_id": call["id"], "content": "739281"}]
            reply, _ = q.completion(url, {"messages": messages})
            self.assertEqual(reply["content"], "739281")
            self.assertEqual(bodies[-1]["messages"], messages)
            with self.assertRaisesRegex(RuntimeError, "without \\[DONE\\]"):
                q.completion(url, {"truncate": True}, True)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
