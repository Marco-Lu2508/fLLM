#!/usr/bin/env python3
"""Offline SSE regression tests; --mutations also proves the assertions bite.

    python3 -B scripts/colab/test_compare_measure.py -v
    python3 -B scripts/colab/test_compare_measure.py --mutations

No model, downloads or generated fixture files. Mutants run in memory only.
"""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, mock_open, patch


SOURCE = Path(__file__).with_name("compare.py")
spec = importlib.util.spec_from_file_location("compare_measure", SOURCE)
assert spec is not None and spec.loader is not None
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


def event(delta=None, usage=None, finish=None):
    choices = [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish}]
    return {"choices": choices, "usage": usage}


def fixture(*events, done=True, compact=False):
    separators = (",", ":") if compact else None
    data = b": heartbeat\r\n\r\n"
    for item in events:
        data += b"data: " + json.dumps(item, separators=separators, ensure_ascii=False).encode() + b"\r\n\r\n"
    return data + (b"data: [DONE]\r\n\r\n" if done else b"")


def counts(prompt=123, completion=11, cached=17):
    return {"prompt_tokens": prompt, "completion_tokens": completion,
            "prompt_tokens_details": {"cached_tokens": cached}}


class StreamTests(unittest.TestCase):
    def request(self, data, extra=None, seed=42):
        sent = []

        def open_fixture(req, **kwargs):
            sent.append(json.loads(req.data))
            return io.BytesIO(data)

        with patch.object(compare.urllib.request, "urlopen", side_effect=open_fixture), \
                patch.object(compare.time, "perf_counter", side_effect=[10.0, 10.5, 12.0]):
            out = []
            compare.one_request("http://fixture/v1/chat/completions", "m", 100, 64, extra or {}, out, seed=seed)
        self.assertEqual(len(out), 1)
        return out[0], sent[0]

    def test_usage_not_chunk_or_word_count(self):
        for compact in (False, True):
            with self.subTest(compact=compact):
                result, body = self.request(fixture(
                    event({"role": "assistant", "content": ""}),
                    event({"content": "one delta can contain many tokenizer tokens"}),
                    event({"content": " and another"}), event({}, finish="stop"),
                    event(usage=counts()), compact=compact))
                self.assertNotIn("error", result)
                self.assertEqual(result["chunks"], 2)
                self.assertEqual(result["completion_tokens"], 11)
                self.assertEqual(result["prompt_tokens"], 123)
                self.assertEqual(result["cached_tokens"], 17)
                self.assertEqual(result["token_source"], "usage")
                self.assertEqual(result["first"], 0.5)
                self.assertEqual(result["total"], 2.0)
                self.assertEqual(body["stream_options"]["include_usage"], True)
                self.assertEqual(body["seed"], 42)
                self.assertIn("Run 42:", body["messages"][0]["content"])

    def test_instrument_options_survive_extras(self):
        _, body = self.request(fixture(event({"content": "ok"}), event(usage=counts())),
                               extra={"stream_options": {"include_usage": False, "other": True},
                                      "stream": False, "seed": 999})
        self.assertTrue(body["stream"])
        self.assertTrue(body["stream_options"]["include_usage"])
        self.assertTrue(body["stream_options"]["other"])
        self.assertEqual(body["seed"], 42)

    def test_reasoning_empty_deltas_and_sample(self):
        result, _ = self.request(fixture(event({"role": "assistant"}), event({"content": None}),
                                         event({"reasoning_content": "think "}),
                                         event({"reasoning": "more "}), event({"content": "answer"}),
                                         event({}, finish="length"), event(usage=counts())))
        self.assertEqual(result["chunks"], 3)
        self.assertEqual(result["first"], 0.5)
        self.assertEqual(result["sample"], "think more answer")

    def test_usage_only_zero_is_valid_and_no_fake_first_token(self):
        result, _ = self.request(fixture(event(usage=counts(prompt=25, completion=0, cached=0))))
        self.assertNotIn("error", result)
        self.assertEqual(result["completion_tokens"], 0)
        self.assertEqual(result["cached_tokens"], 0)
        self.assertEqual(result["token_source"], "usage")
        self.assertIsNone(result["first"])
        row = compare.summarize_requests(100, 1, [result], 2)
        self.assertEqual(row["total_tok_s"], 0.0)
        self.assertIsNone(row["first_s"])

    def test_cumulative_usage_is_not_summed(self):
        result, _ = self.request(fixture(event({"content": "a"}, counts(completion=2)),
                                         event({"content": "b"}, counts(completion=5)),
                                         event(usage=counts(completion=9))))
        self.assertEqual(result["completion_tokens"], 9)

    def test_missing_usage_is_explicit_chunk_fallback(self):
        result, _ = self.request(fixture(event({"content": "many tokens"}), event({}, finish="stop")))
        self.assertIsNone(result["completion_tokens"])
        self.assertIsNone(result["prompt_tokens"])
        self.assertIsNone(result["cached_tokens"])
        self.assertEqual(result["token_source"], "sse_chunks_unverified")
        self.assertIn("not tokens", result["measurement_warning"])
        row = compare.summarize_requests(100, 1, [result], 2)
        self.assertIsNone(row["total_tok_s"])
        self.assertIsNone(row["each_tok_s"])
        self.assertEqual(row["total_chunks_s"], 0.5)

    def test_partial_usage_preserves_prompt_and_unknown_cache(self):
        result, _ = self.request(fixture(event({"content": "a"}), event(usage={"prompt_tokens": 27})))
        self.assertEqual(result["prompt_tokens"], 27)
        self.assertIsNone(result["completion_tokens"])
        self.assertIsNone(result["cached_tokens"])
        result, _ = self.request(fixture(event({"content": "a"}),
                                         event(usage={"prompt_tokens": 27, "completion_tokens": 4})))
        self.assertEqual(result["token_source"], "usage")
        self.assertIsNone(result["cached_tokens"])

    def test_invalid_counts_are_errors(self):
        for name in ("prompt_tokens", "completion_tokens"):
            for invalid in (-1, True, 1.5, "3"):
                with self.subTest(name=name, invalid=invalid):
                    usage = counts()
                    usage[name] = invalid
                    result, _ = self.request(fixture(event({"content": "a"}), event(usage=usage)))
                    self.assertIn("invalid usage count", result["error"])
        for cached in (-1, True, 124):
            result, _ = self.request(fixture(event({"content": "a"}), event(usage=counts(cached=cached))))
            self.assertIn("error", result)

    def test_stream_errors_malformed_json_and_truncation_are_not_success(self):
        for data, expected in ((fixture(event({"content": "partial"}), done=False), "missing [DONE]"),
                               (fixture({"error": {"message": "worker failed"}}), "worker failed"),
                               (b"data: {not json}\n\ndata: [DONE]\n\n", "Expecting"),
                               (b"", "missing [DONE]")):
            with self.subTest(expected=expected):
                result, _ = self.request(data)
                self.assertIn("error", result)
                self.assertIn(expected, result["error"])

    def test_multiline_sse_and_utf8(self):
        data = ('event: message\r\nid: 1\r\ndata: {"choices":\r\n'
                'data: [{"delta": {"content": "Grüße"}}]}\r\n\r\n'
                'data: [DONE]\r\n\r\n').encode()
        result, _ = self.request(data)
        self.assertEqual(result["chunks"], 1)
        self.assertEqual(result["sample"], "Grüße")

    def test_sample_is_capped(self):
        result, _ = self.request(fixture(event({"content": "x" * 300}), event(usage=counts())))
        self.assertEqual(len(result["sample"]), 160)

    def test_http_error_is_recorded(self):
        out = []
        with patch.object(compare.urllib.request, "urlopen", side_effect=OSError("fixture offline")):
            compare.one_request("http://fixture", "m", 100, 64, {}, out, seed=5)
        self.assertEqual(out, [{"error": "fixture offline", "seed": 5}])


class MeasurementTests(unittest.TestCase):
    def sample(self, seed, made: int | None = 11, prompt: int | None = 123,
               cached: int | None = 17, source="usage"):
        return {"seed": seed, "first": 0.5, "total": 2.0, "chunks": 2, "sample": "ok",
                "completion_tokens": made, "prompt_tokens": prompt, "cached_tokens": cached,
                "token_source": source}

    def test_real_token_aggregation_and_e2e_not_fake_post_first_rate(self):
        out = [self.sample(2, 15, 125, 3), self.sample(1)]
        row = compare.summarize_requests(100, 2, out, 4)
        self.assertEqual(row["total_tok_s"], 6.5)  # 26 actual tokens / 4s, not 4 chunks / 4s
        self.assertEqual(row["each_tok_s"], 6.5)  # mean(11/2, 15/2), NOT (tokens-1)/(total-first)
        self.assertEqual(row["each_tok_s_basis"], "end_to_end_including_ttft")
        self.assertEqual(row["actual_prompt_tokens"], [123, 125])
        self.assertEqual(row["prompt_tokens"], 248)
        self.assertEqual(row["completion_tokens"], 26)
        self.assertEqual(row["cached_tokens"], 20)
        self.assertEqual(row["token_source"], "usage")
        self.assertEqual(row["total_chunks_s"], 1.0)
        self.assertEqual(row["first_s"], 0.5)
        self.assertEqual([o["seed"] for o in row["requests"]], [1, 2])
        json.dumps(row, allow_nan=False)

    def test_mixed_sources_do_not_publish_partial_token_rate(self):
        out = [self.sample(1), self.sample(2, made=None, prompt=None, cached=None, source="sse_chunks_unverified")]
        row = compare.summarize_requests(100, 2, out, 4)
        self.assertEqual(row["token_source"], "mixed")
        for name in ("total_tok_s", "each_tok_s", "completion_tokens", "prompt_tokens", "cached_tokens"):
            self.assertIsNone(row[name])
        self.assertIn("not tok/s", row["measurement_warning"])

    def test_failed_request_invalidates_batch(self):
        row = compare.summarize_requests(100, 2, [self.sample(1), {"seed": 2, "error": "failure"}], 4)
        self.assertEqual(row["error"], "failure")
        self.assertNotIn("total_tok_s", row)

    def test_missing_thread_result_invalidates_batch(self):
        for out in ([], [self.sample(1)]):
            row = compare.summarize_requests(100, 2, out, 4)
            self.assertIn("request results for 2 users", row["error"])
            self.assertNotIn("total_tok_s", row)

    def test_deterministic_ab_requests_and_unique_nonces(self):
        def run(model, seed=123):
            sent = []

            def open_fixture(req, **kwargs):
                sent.append(json.loads(req.data))
                return io.BytesIO(fixture(event({"content": "ok"}), event(usage=counts())))

            with patch.object(compare.urllib.request, "urlopen", side_effect=open_fixture), \
                    patch.object(compare, "log"):
                rows = compare.measure(types.SimpleNamespace(model=model), 1234, [100, 500], [1, 4], 64, {}, seed=seed)
            return sorted(sent, key=lambda b: (b["seed"], b["max_tokens"])), rows

        a, rows = run("engine-A")
        b, _ = run("engine-B")
        for body in a + b:
            body.pop("model")
        self.assertEqual(a, b)
        c, _ = run("engine-C", seed=124)
        for body in c:
            body.pop("model")
        self.assertNotEqual(a, c)
        self.assertEqual(len(rows), 5)
        for row in rows[1:]:
            self.assertEqual(len(row["requests"]), row["users"])
            seeds = [o["seed"] for o in row["requests"]]
            self.assertEqual(len(seeds), len(set(seeds)))
            self.assertTrue(all(0 <= s < 2**31 for s in seeds))
            self.assertEqual(len(row["actual_prompt_tokens"]), row["users"])

    def test_control_disables_all_three_accelerators(self):
        args, env = compare.fllm_control_options('--chunked-prefill-policy "prefill-first"')
        self.assertEqual(args, ["--chunked-prefill-policy", "prefill-first", "--speculative", "off"])
        self.assertEqual(env["FLLM_PREFIX_CACHE"], "off")
        self.assertEqual(env["FLLM_OUTPUT_CACHE"], "off")
        self.assertEqual(env["FLLM_SPECULATIVE_DEFAULT"], "off")
        self.assertEqual(env["FLLM_HEAD_FULL"], "1")  # the same 16-bit output projection as the rival engines, not fLLM's Q6_K copy of it
        for option in ("--speculative suffix", "--speculative=ngram", "--speculative-model draft.gguf",
                       "--speculative-lookup 4", "-sp 4", "-sw 4", "--num-speculative-tokens 4"):
            with self.subTest(option=option), self.assertRaisesRegex(ValueError, "cannot override"):
                compare.fllm_control_options(option)

    def test_main_wires_control_seed_json_and_unavailable_table(self):
        server = types.SimpleNamespace(ok=True, model="m", proc=Mock(), stop=Mock())
        server.proc.poll.return_value = None
        result = {}
        output = io.StringIO()
        argv = ["compare.py", "--cpu", "--fllm", "/fixture/fllm", "--work", "/fixture/work",
                "--models", "llama8b", "--sizes", "100", "--users", "1", "--seed", "123",
                "--no-llama", "--no-ik", "--no-ollama", "--no-vllm", "--no-sglang"]

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", argv))
            stack.enter_context(patch.object(compare.os, "makedirs"))
            stack.enter_context(patch.object(compare.os, "chdir"))
            stack.enter_context(patch.object(compare.os.path, "exists", return_value=True))
            stack.enter_context(patch.object(compare, "get_fllm", return_value="/fixture/fllm"))
            stack.enter_context(patch.object(compare, "download", return_value="/fixture/model.gguf"))
            stack.enter_context(patch.object(compare, "sh", return_value=(0, "fixture CPU/version")))
            start = stack.enter_context(patch.object(compare, "Server", return_value=server))
            stack.enter_context(patch.object(compare.time, "sleep"))
            stack.enter_context(patch("builtins.open", mock_open()))
            stack.enter_context(patch.object(compare.json, "dump", side_effect=lambda obj, *a, **k: result.update(obj)))
            stack.enter_context(patch.object(compare.urllib.request, "urlopen",
                                            side_effect=lambda *a, **k: io.BytesIO(fixture(event({"content": "ok"})))))
            stack.enter_context(contextlib.redirect_stdout(output))
            compare.main()

        self.assertEqual(result["seed"], 123)
        self.assertEqual(result["schema_version"], 2)
        self.assertEqual(result["measurement"]["each_tok_s"], "end_to_end_including_ttft")
        self.assertEqual(start.call_args.args[0][-2:], ["--speculative", "off"])
        self.assertIn("env", start.call_args.kwargs)
        self.assertEqual(start.call_args.kwargs["env"], result["fllm_control"]["env"])
        self.assertEqual(start.call_args.kwargs["env"]["FLLM_OUTPUT_CACHE"], "off")
        self.assertEqual(start.call_args.kwargs["env"]["FLLM_PREFIX_CACHE"], "off")
        rows = next(iter(result["models"].values()))["fllm"]
        self.assertEqual(rows[1]["requests"][0]["seed"], compare.request_seed(123, 100, 1, 0))
        self.assertIsNone(rows[1]["total_tok_s"])
        self.assertIn("warning fllm", output.getvalue())
        self.assertIn("- |", output.getvalue())
        server.stop.assert_called_once()

    def test_server_controls_override_inherited_environment(self):
        _, control = compare.fllm_control_options("")
        inherited = {name: "on" for name in control}
        with patch.dict(compare.os.environ, inherited), patch.object(compare.subprocess, "Popen") as popen, \
                patch.object(compare, "wait_health", return_value=True), patch("builtins.open", mock_open()):
            compare.Server(["fixture"], "http://fixture/health", "m", env=control)
        for name, value in control.items():
            self.assertEqual(popen.call_args.kwargs["env"][name], value)


class BestRivalTests(unittest.TestCase):
    """The yardstick of a cell is the best rival, not the weakest one."""

    @staticmethod
    def table(**rates):
        return {name: [{"total_tok_s": rate}] for name, rate in rates.items()}

    def test_ratio_is_against_the_fastest_rival(self):
        table = self.table(fllm=60.0, **{"llama.cpp": 40.0, "ik_llama.cpp": 80.0, "vllm": 50.0})
        got = compare.best_ratio(table, list(table), 0)
        self.assertEqual(got["best_rival"], "ik_llama.cpp")
        self.assertEqual(got["ratio_vs_best"], 0.75)
        self.assertEqual(got["best_rival_tok_s"], 80.0)

    def test_a_win_over_the_best_rival_is_above_one(self):
        table = self.table(fllm=100.0, vllm=80.0, sglang=90.0)
        self.assertEqual(compare.best_ratio(table, list(table), 0)["ratio_vs_best"], 1.11)

    def test_failed_or_unavailable_rivals_do_not_count(self):
        table = {"fllm": [{"total_tok_s": 10.0}], "vllm": [{"error": "did not start"}], "llama.cpp": [{"total_tok_s": None}]}
        self.assertIsNone(compare.best_ratio(table, list(table), 0))

    def test_no_fllm_figure_gives_no_ratio(self):
        table = self.table(fllm=None, vllm=50.0)
        self.assertIsNone(compare.best_ratio(table, list(table), 0))
        self.assertIsNone(compare.best_ratio(self.table(vllm=50.0), ["vllm"], 0))


class EngineVersionTests(unittest.TestCase):
    def test_every_installed_rival_reports_its_version_and_missing_ones_are_left_out(self):
        calls = []

        def fake(cmd, timeout=None, env=None):
            calls.append(cmd)
            return 0, "banner line\n9.8.7\n"

        with patch.object(compare, "sh", side_effect=fake):
            got = compare.engine_versions({"fllm": "/b/fllm-rs", "vllm": "/env/bin/vllm", "sglang": "/sg/bin/python",
                                           "llama.cpp": "/l/llama-server", "ik_llama.cpp": None, "ollama": "/usr/bin/ollama"})
        self.assertEqual(got, {"vllm": "9.8.7", "sglang": "9.8.7", "llama.cpp": "9.8.7"})
        self.assertTrue(any("/env/bin/python -c" in c and "vllm" in c for c in calls))
        self.assertTrue(any(c.startswith("/sg/bin/python -c") and "sglang" in c for c in calls))

    def test_a_failing_version_probe_is_not_invented(self):
        with patch.object(compare, "sh", return_value=(1, "boom")):
            self.assertEqual(compare.engine_versions({"vllm": "/env/bin/vllm"}), {})


def run_mutations():
    """Compile valid mutants without touching the source or creating extra files."""
    source = SOURCE.read_text()
    mutations = [
        ("usage replaced by chunks", 'completion = usage_count(usage, "completion_tokens")', 'completion = chunks'),
        ("include_usage disabled", '"include_usage": True}', '"include_usage": False}'),
        ("prompt count fabricated", 'prompt = usage_count(usage, "prompt_tokens")', 'prompt = size'),
        ("cache count fabricated", 'cached = usage_count(details, "cached_tokens")', 'cached = 0'),
        ("unknown usage sold as tok/s", 'round(row["completion_tokens"] / wall, 1) if measured and wall > 0 else None',
         'round(sum(o["chunks"] for o in out) / wall, 1) if wall > 0 else None'),
        ("request seeds ignored", 'body["seed"] = seed', 'body["seed"] = 999'),
        ("AB seeds randomized", 'request_seed(seed, size, n, i)))', 'random.randint(0, 10**9)))'),
        ("prefix cache enabled", '"FLLM_PREFIX_CACHE": "off"', '"FLLM_PREFIX_CACHE": "on"'),
        ("output cache enabled", '"FLLM_OUTPUT_CACHE": "off"', '"FLLM_OUTPUT_CACHE": "exact"'),
        ("speculation enabled", 'args + ["--speculative", "off"]', 'args + ["--speculative", "suffix"]'),
        ("control environment not passed to server", '"m", env=control_env, logfile=f"fllm-{key}.log"',
         '"m", logfile=f"fllm-{key}.log"'),
        ("batch seed not passed from CLI", 'extra, seed=args.seed)', 'extra, seed=0)'),
        ("truncated streams accepted", 'if not done:', 'if False:'),
        ("version probe result invented", 'if rc == 0 and text.strip():', 'if True:'),
        ("vllm probed with the wrong interpreter", "os.path.join(os.path.dirname(binary), 'python')", "'python3'"),
        ("weakest rival as yardstick", 'best = max(rivals, key=rivals.get)', 'best = min(rivals, key=rivals.get)'),
        ("ratio inverted", '"ratio_vs_best": round(mine / rivals[best], 2)', '"ratio_vs_best": round(rivals[best] / mine, 2)'),
        ("failed rival counted", 'return r.get("total_tok_s") if "error" not in r else None', 'return r.get("total_tok_s") or 1.0e9'),
    ]
    killed = 0
    for name, old, new in mutations:
        if source.count(old) != 1:
            raise AssertionError(f"mutation anchor {name!r} occurs {source.count(old)} times")
        mutant = types.ModuleType("compare_mutant")
        exec(compile(source.replace(old, new, 1), str(SOURCE), "exec"), mutant.__dict__)
        stream = io.StringIO()
        suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls)
                                   for cls in (StreamTests, MeasurementTests, BestRivalTests, EngineVersionTests))
        with patch.object(sys.modules[__name__], "compare", mutant), contextlib.redirect_stdout(io.StringIO()):
            result = unittest.TextTestRunner(stream=stream).run(suite)
        # A thrown exception alone does not prove an assertion detects the regression.
        detected = bool(result.failures)
        killed += int(detected)
        print(f"{'KILLED' if detected else 'SURVIVED'}: {name} "
              f"({len(result.failures)} assertion failures, {len(result.errors)} errors)")
        if not detected:
            print(stream.getvalue())
    print(f"Mutation result: {killed}/{len(mutations)} killed (in-memory; no source edits).")
    return killed == len(mutations)


if __name__ == "__main__":
    if "--mutations" in sys.argv:
        sys.exit(0 if run_mutations() else 1)
    unittest.main()
