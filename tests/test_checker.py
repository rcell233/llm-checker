import json
import os
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import add_model
import llm_checker

ROOT = Path(__file__).resolve().parents[1]


class CheckerTests(unittest.TestCase):
    def test_stdlib_scoring_matches_reference_implementation(self):
        import fingerprint

        bank = json.loads((ROOT / "data/unified_bank.json").read_text())
        for file in ("gpt_reference.jsonl", "claude_reference.jsonl"):
            rows = (ROOT / "data" / file).read_text().splitlines()
            for line in (rows[0], rows[len(rows) // 2], rows[-1]):
                row = json.loads(line)
                numbers = fingerprint.parse_numbers(row["text"])
                expected = fingerprint.robust_score_numbers(numbers, bank)["fused"]
                actual = llm_checker.score(numbers, bank)
                self.assertEqual(len(actual), len(expected))
                for left, right in zip(actual, expected):
                    self.assertAlmostEqual(left, right, places=12)

    def test_one_probe_cli_path(self):
        bank = json.loads((ROOT / "data/unified_bank.json").read_text())
        row = json.loads((ROOT / "data/gpt_reference.jsonl").read_text().splitlines()[0])
        with patch.object(llm_checker, "load_bank", return_value=bank), \
             patch.object(llm_checker, "codex_executable", return_value="codex"), \
             patch.object(llm_checker, "run_codex", return_value=(row["text"], {"output_tokens": 10})) as run:
            self.assertEqual(llm_checker.main(["-m", "gpt-5.5", "-n", "1"]), 0)
            self.assertEqual(run.call_args.args[2], "low")

    def test_model_collection_resumes_without_duplicate_rows(self):
        from challenge_suite import fingerprint_suite

        with tempfile.TemporaryDirectory() as directory:
            tasks = fingerprint_suite()[:3]
            command = ["--label", "test-model", "--family", "test", "--api-model", "test-model",
                       "--base-url", "https://example.invalid/v1", "-n", "3", "--no-rebuild"]
            with patch.object(add_model, "DATA", Path(directory)), \
                 patch.object(add_model, "fingerprint_suite", return_value=tasks), \
                 patch.object(add_model, "request_completion", return_value=" ".join(["17"] * 355)) as request, \
                 patch.dict(os.environ, {"OPENAI_API_KEY": "test-secret"}):
                self.assertEqual(add_model.main(command), 0)
                self.assertEqual(add_model.main(command), 0)
            rows = (Path(directory) / "test_reference.jsonl").read_text().splitlines()
            self.assertEqual(len(rows), 3)
            self.assertEqual(request.call_count, 3)
            self.assertTrue(all(json.loads(line)["strict_valid"] for line in rows))

    def test_defaults_use_three_probes_and_low_reasoning(self):
        bank = json.loads((ROOT / "data/unified_bank.json").read_text())
        row = json.loads((ROOT / "data/gpt_reference.jsonl").read_text().splitlines()[0])
        probes = [{"id": str(index), "expected_count": 218, "prompt": "test"} for index in range(3)]
        with patch.object(llm_checker, "load_bank", return_value=bank), \
             patch.object(llm_checker, "codex_executable", return_value="codex"), \
             patch.object(llm_checker, "challenges", return_value=probes) as generate, \
             patch.object(llm_checker, "run_codex", return_value=(row["text"], {})) as run:
            self.assertEqual(llm_checker.main(["-m", "gpt-6-astra"]), 0)
        generate.assert_called_once_with(3)
        self.assertEqual(run.call_count, 3)
        self.assertTrue(all(call.args[2] == "low" for call in run.call_args_list))

    def test_three_probes_start_concurrently_when_requested(self):
        bank = json.loads((ROOT / "data/unified_bank.json").read_text())
        row = json.loads((ROOT / "data/gpt_reference.jsonl").read_text().splitlines()[0])
        probes = [{"id": str(index), "expected_count": 218, "prompt": str(index)} for index in range(3)]
        barrier = threading.Barrier(3)

        def run(*_):
            barrier.wait(timeout=2)
            return row["text"], {}

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            with patch.object(llm_checker, "load_bank", return_value=bank), \
                 patch.object(llm_checker, "codex_executable", return_value="codex"), \
                 patch.object(llm_checker, "challenges", return_value=probes), \
                 patch.object(llm_checker, "run_codex", side_effect=run):
                self.assertEqual(llm_checker.main(["-j", "3", "--output", str(output)]), 0)
            saved = json.loads(output.read_text())
            self.assertEqual([item["id"] for item in saved["responses"]], ["0", "1", "2"])

    def test_default_one_worker_runs_probes_serially(self):
        bank = json.loads((ROOT / "data/unified_bank.json").read_text())
        row = json.loads((ROOT / "data/gpt_reference.jsonl").read_text().splitlines()[0])
        probes = [{"id": str(index), "expected_count": 218, "prompt": str(index)} for index in range(3)]
        lock = threading.Lock()
        active = peak = 0

        def run(*_):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.01)
            with lock:
                active -= 1
            return row["text"], {}

        with patch.object(llm_checker, "load_bank", return_value=bank), \
             patch.object(llm_checker, "codex_executable", return_value="codex"), \
             patch.object(llm_checker, "challenges", return_value=probes), \
             patch.object(llm_checker, "run_codex", side_effect=run):
            self.assertEqual(llm_checker.main(["-j", "1"]), 0)
        self.assertEqual(peak, 1)

    def test_account_info_reads_custom_provider_from_config(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.toml").write_text(
                'model_provider = "sky_router"\n'
                "[model_providers.sky_router]\n"
                'base_url = "https://example.invalid/v1"\n'
                'name = "SkyRouter"\n'
                "[model_providers.sky_router.http_headers]\n"
                'Authorization = "Bearer sk-ABCDEFGHabcdefgh"\n',
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"CODEX_HOME": directory}):
                info = llm_checker.get_codex_account_info()
        self.assertEqual(info, "API: SkyRouter (example.invalid, sk-A...efgh)")

    def test_account_info_uses_provider_table_when_auth_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.toml").write_text(
                "[model_providers.sky_router]\n"
                'base_url = "https://example.invalid/v1"\n'
                'name = "SkyRouter"\n'
                "[model_providers.sky_router.http_headers]\n"
                'Authorization = "Bearer sk-ABCDEFGHabcdefgh"\n',
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"CODEX_HOME": directory}):
                info = llm_checker.get_codex_account_info()
        self.assertIn("SkyRouter", info)
        self.assertIn("sk-A...efgh", info)
        self.assertNotEqual(info, "本地 Codex")

    def test_account_info_reads_env_key(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.toml").write_text(
                'model_provider = "proxy"\n'
                "[model_providers.proxy]\n"
                'name = "Proxy"\n'
                'base_url = "https://example.invalid/v1"\n'
                'env_key = "MY_PROVIDER_KEY"\n',
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"CODEX_HOME": directory, "MY_PROVIDER_KEY": "sk-ENVKEY12345678"}):
                info = llm_checker.get_codex_account_info()
        self.assertEqual(info, "API: Proxy (example.invalid, sk-E...5678)")

    def test_account_info_reads_env_http_headers(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.toml").write_text(
                'model_provider = "proxy"\n'
                "[model_providers.proxy]\n"
                'name = "Proxy"\n'
                'base_url = "https://example.invalid/v1"\n'
                "[model_providers.proxy.env_http_headers]\n"
                'Authorization = "MY_AUTH_HEADER"\n',
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"CODEX_HOME": directory, "MY_AUTH_HEADER": "Bearer sk-HEADERKEY1234"}):
                info = llm_checker.get_codex_account_info()
        self.assertEqual(info, "API: Proxy (example.invalid, sk-H...1234)")

    def test_account_info_reads_experimental_bearer_token(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.toml").write_text(
                'model_provider = "proxy"\n'
                "[model_providers.proxy]\n"
                'name = "Proxy"\n'
                'base_url = "https://example.invalid/v1"\n'
                'experimental_bearer_token = "sk-BEARERTOKEN1234"\n',
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"CODEX_HOME": directory}):
                info = llm_checker.get_codex_account_info()
        self.assertEqual(info, "API: Proxy (example.invalid, sk-B...1234)")

    def test_account_info_reads_auth_command(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.toml").write_text(
                'model_provider = "proxy"\n'
                "[model_providers.proxy]\n"
                'name = "Proxy"\n'
                'base_url = "https://example.invalid/v1"\n'
                "[model_providers.proxy.auth]\n"
                'command = "/usr/local/bin/fetch-codex-token"\n',
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"CODEX_HOME": directory}):
                info = llm_checker.get_codex_account_info()
        self.assertEqual(info, "API: Proxy (example.invalid, 凭证命令 fetch-codex-token)")

    def test_account_info_reads_openai_api_key_env(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"CODEX_HOME": directory, "OPENAI_API_KEY": "sk-OPENAIKEY123456"}):
                info = llm_checker.get_codex_account_info()
        self.assertEqual(info, "API Key: sk-O...3456（环境变量）")

    def test_account_info_reads_auth_json_api_key(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "auth.json").write_text(json.dumps({
                "auth_mode": "api",
                "OPENAI_API_KEY": "sk-AUTHJSON12345678",
            }), encoding="utf-8")
            with patch.dict(os.environ, {"CODEX_HOME": directory}):
                info = llm_checker.get_codex_account_info()
        self.assertEqual(info, "API Key: sk-A...5678")

    def test_cli_prints_codex_home(self):
        import io

        bank = json.loads((ROOT / "data/unified_bank.json").read_text())
        row = json.loads((ROOT / "data/gpt_reference.jsonl").read_text().splitlines()[0])
        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(llm_checker, "load_bank", return_value=bank), \
                 patch.object(llm_checker, "codex_executable", return_value="codex"), \
                 patch.object(llm_checker, "run_codex", return_value=(row["text"], {})), \
                 patch.dict(os.environ, {"CODEX_HOME": directory}), \
                 patch("sys.stdout", stdout):
                self.assertEqual(llm_checker.main(["-n", "1"]), 0)
        output = stdout.getvalue()
        self.assertIn(f"CODEX_HOME：{directory}（环境变量）", output)
        self.assertIn("Codex 账号：", output)

    def test_defaults_use_one_hundred_twenty_second_timeout(self):
        bank = json.loads((ROOT / "data/unified_bank.json").read_text())
        row = json.loads((ROOT / "data/gpt_reference.jsonl").read_text().splitlines()[0])
        with patch.object(llm_checker, "load_bank", return_value=bank), \
             patch.object(llm_checker, "codex_executable", return_value="codex"), \
             patch.object(llm_checker, "run_codex", return_value=(row["text"], {})) as run:
            self.assertEqual(llm_checker.main(["-m", "gpt-5.5", "-n", "1"]), 0)
            self.assertEqual(run.call_args.args[4], 120)

    def test_account_info_prefers_chatgpt_when_provider_is_not_selected(self):
        import base64

        payload = base64.urlsafe_b64encode(json.dumps({"email": "user@example.com"}).encode()).decode().rstrip("=")
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.toml").write_text(
                "[model_providers.sky_router]\n"
                'base_url = "https://example.invalid/v1"\n'
                'name = "SkyRouter"\n',
                encoding="utf-8",
            )
            Path(directory, "auth.json").write_text(json.dumps({
                "auth_mode": "chatgpt",
                "tokens": {"id_token": f"header.{payload}.sig"},
            }), encoding="utf-8")
            with patch.dict(os.environ, {"CODEX_HOME": directory}):
                info = llm_checker.get_codex_account_info()
        self.assertEqual(info, "ChatGPT 账号: user@example.com")

    def test_run_codex_times_out(self):
        with patch.object(llm_checker.subprocess, "run", side_effect=subprocess.TimeoutExpired(cmd="codex", timeout=1)):
            with self.assertRaisesRegex(RuntimeError, "超过 1 秒未返回"):
                llm_checker.run_codex("codex", "m", "low", "prompt", timeout=1)

    def test_run_codex_rejects_repeat_loop_answer(self):
        payload = json.dumps({
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "1, " * 5000},
        })
        completed = Mock(returncode=0, stdout=payload + "\n", stderr="")
        with patch.object(llm_checker.subprocess, "run", return_value=completed):
            with self.assertRaisesRegex(RuntimeError, "回答过长"):
                llm_checker.run_codex("codex", "m", "low", "prompt")

    def test_cli_passes_timeout_to_codex(self):
        bank = json.loads((ROOT / "data/unified_bank.json").read_text())
        row = json.loads((ROOT / "data/gpt_reference.jsonl").read_text().splitlines()[0])
        with patch.object(llm_checker, "load_bank", return_value=bank), \
             patch.object(llm_checker, "codex_executable", return_value="codex"), \
             patch.object(llm_checker, "run_codex", return_value=(row["text"], {})) as run:
            self.assertEqual(llm_checker.main(["-m", "gpt-5.5", "-n", "1", "--timeout", "15"]), 0)
            self.assertEqual(run.call_args.args[4], 15)


if __name__ == "__main__":
    unittest.main()
