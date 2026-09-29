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



class ClaudeConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.home = root / "home"
        self.project = root / "project"
        self.managed = root / "managed"
        for path in (self.home / ".claude", self.project / ".claude", self.managed):
            path.mkdir(parents=True)
        self.env = patch.dict(os.environ, {"HOME": str(self.home), "PATH": os.environ.get("PATH", "")}, clear=True)
        self.env.start()
        self.managed_patch = patch.object(llm_checker, "_managed_settings_dir", return_value=self.managed)
        self.managed_patch.start()

    def tearDown(self):
        self.managed_patch.stop()
        self.env.stop()
        self.temp.cleanup()

    def write(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    def resolve(self, tier="opus"):
        return llm_checker.resolve_claude_config(tier, self.project)

    def test_tier_detection(self):
        self.assertEqual(llm_checker.claude_tier(" Opus "), "opus")
        self.assertEqual(llm_checker.claude_tier("fable"), "fable")
        self.assertEqual(llm_checker.claude_tier("gpt-5.5"), "")
        self.assertEqual(llm_checker.claude_tier(None), "")

    def test_settings_layers_follow_claude_code_precedence(self):
        os.environ["ANTHROPIC_BASE_URL"] = "https://shell.invalid"
        self.write(self.home / ".claude/settings.json", {"env": {
            "ANTHROPIC_BASE_URL": "https://user.invalid/",
            "ANTHROPIC_AUTH_TOKEN": "sk-USERTOKEN12345678",
            "ANTHROPIC_DEFAULT_OPUS_MODEL": "claude-opus-user",
            "ANTHROPIC_DEFAULT_SONNET_MODEL": "claude-sonnet-user",
        }})
        self.write(self.project / ".claude/settings.json", {"env": {
            "ANTHROPIC_DEFAULT_OPUS_MODEL": "claude-opus-project",
            "ANTHROPIC_BASE_URL": "https://project.invalid",
        }})
        self.write(self.project / ".claude/settings.local.json", {"env": {"ANTHROPIC_DEFAULT_OPUS_MODEL": "claude-opus-local"}})
        config = self.resolve()
        self.assertEqual(config["model"], "claude-opus-local")
        self.assertEqual(config["model_var"], "ANTHROPIC_DEFAULT_OPUS_MODEL")
        self.assertTrue(config["model_source"].endswith(".claude/settings.local.json"))
        self.assertEqual(config["url"], "https://project.invalid/v1/messages")
        self.assertEqual(config["headers"]["authorization"], "Bearer sk-USERTOKEN12345678")
        self.assertNotIn("x-api-key", config["headers"])
        self.assertEqual(config["credential_source"], "~/.claude/settings.json")
        self.assertEqual(self.resolve("sonnet")["model"], "claude-sonnet-user")
        self.assertEqual(len(config["files"]), 3)

    def test_managed_settings_and_drop_ins_override_everything(self):
        os.environ["ANTHROPIC_API_KEY"] = "sk-SHELLKEY12345678"
        self.write(self.project / ".claude/settings.local.json", {"env": {"ANTHROPIC_DEFAULT_OPUS_MODEL": "local"}})
        self.write(self.managed / "managed-settings.json", {"env": {"ANTHROPIC_DEFAULT_OPUS_MODEL": "managed"}})
        self.write(self.managed / "managed-settings.d/10-a.json", {"env": {"ANTHROPIC_DEFAULT_OPUS_MODEL": "drop-in-a"}})
        self.write(self.managed / "managed-settings.d/20-b.json", {"env": {"ANTHROPIC_DEFAULT_OPUS_MODEL": "drop-in-b"}})
        config = self.resolve()
        self.assertEqual(config["model"], "drop-in-b")
        self.assertEqual(config["headers"]["x-api-key"], "sk-SHELLKEY12345678")
        self.assertEqual(config["credential_source"], "环境变量")
        self.assertEqual(config["url"], "https://api.anthropic.com/v1/messages")

    def test_claude_config_dir_relocates_user_settings(self):
        custom = Path(self.temp.name) / "custom-claude"
        self.write(custom / "settings.json", {"env": {"ANTHROPIC_API_KEY": "sk-CUSTOMDIR1234567"}})
        self.write(self.home / ".claude/settings.json", {"env": {"ANTHROPIC_API_KEY": "sk-IGNORED123456789"}})
        os.environ["CLAUDE_CONFIG_DIR"] = str(custom)
        config = self.resolve()
        self.assertEqual(config["secret"], "sk-CUSTOMDIR1234567")
        self.assertEqual(config["config_dir"], custom)

    def test_model_name_fallback_and_context_suffix(self):
        self.write(self.home / ".claude/settings.json", {"env": {
            "ANTHROPIC_API_KEY": "sk-TESTKEY123456789",
            "ANTHROPIC_DEFAULT_OPUS_MODEL_NAME": "claude-opus-5-5",
            "ANTHROPIC_DEFAULT_FABLE_MODEL": "claude-fable-5-1[1m]",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL_NAME": "Haiku (fast)",
        }})
        opus = self.resolve("opus")
        self.assertEqual(opus["model"], "claude-opus-5-5")
        self.assertEqual(opus["model_var"], "ANTHROPIC_DEFAULT_OPUS_MODEL_NAME")
        self.assertEqual(self.resolve("fable")["model"], "claude-fable-5-1")
        haiku = self.resolve("haiku")
        self.assertEqual(haiku["model"], llm_checker.CLAUDE_DEFAULT_MODELS["haiku"])
        self.assertEqual(haiku["model_var"], "")

    def test_empty_settings_value_unsets_shell_token(self):
        os.environ["ANTHROPIC_AUTH_TOKEN"] = "sk-SHELLTOKEN123456"
        self.write(self.home / ".claude/settings.json", {"env": {
            "ANTHROPIC_AUTH_TOKEN": "", "ANTHROPIC_API_KEY": "sk-SETTINGSKEY12345"}})
        config = self.resolve()
        self.assertEqual(config["credential"], "ANTHROPIC_API_KEY")
        self.assertEqual(config["headers"]["x-api-key"], "sk-SETTINGSKEY12345")
        self.assertNotIn("authorization", config["headers"])

    def test_api_key_helper_output_is_used(self):
        self.write(self.project / ".claude/settings.json", {"apiKeyHelper": "echo sk-HELPERKEY1234567"})
        config = self.resolve()
        self.assertEqual(config["credential"], "apiKeyHelper")
        self.assertEqual(config["headers"]["x-api-key"], "sk-HELPERKEY1234567")
        self.assertEqual(config["headers"]["authorization"], "Bearer sk-HELPERKEY1234567")

    def test_custom_headers_betas_and_proxy_from_settings(self):
        self.write(self.home / ".claude/settings.json", {"env": {
            "ANTHROPIC_AUTH_TOKEN": "sk-TESTTOKEN1234567",
            "ANTHROPIC_BASE_URL": "https://relay.invalid/api",
            "ANTHROPIC_CUSTOM_HEADERS": "X-Relay-Group: team-a\nX-Trace:  on ",
            "ANTHROPIC_BETAS": "beta-a, beta-b",
            "HTTPS_PROXY": "http://127.0.0.1:7890",
        }})
        config = self.resolve()
        self.assertEqual(config["url"], "https://relay.invalid/api/v1/messages")
        self.assertEqual(config["headers"]["x-relay-group"], "team-a")
        self.assertEqual(config["headers"]["x-trace"], "on")
        self.assertEqual(config["headers"]["anthropic-beta"], "beta-a,beta-b")
        self.assertEqual(config["proxies"], {"https": "http://127.0.0.1:7890"})
        os.environ["NO_PROXY"] = "relay.invalid"
        self.assertEqual(self.resolve()["proxies"], {})

    def test_official_login_is_rejected(self):
        (self.home / ".claude/.credentials.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "官方账号"):
            self.resolve()

    def test_missing_credentials_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "未找到 Claude Code 的 API 凭证"):
            self.resolve()

    def test_cloud_provider_is_rejected(self):
        self.write(self.home / ".claude/settings.json", {"env": {
            "CLAUDE_CODE_USE_BEDROCK": "1", "ANTHROPIC_API_KEY": "sk-TESTKEY123456789"}})
        with self.assertRaisesRegex(RuntimeError, "Amazon Bedrock"):
            self.resolve()

    def test_invalid_settings_json_is_reported(self):
        (self.project / ".claude/settings.json").write_text("{", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "不是合法 JSON"):
            self.resolve()

    def test_cli_uses_claude_api_for_tier(self):
        bank = json.loads((ROOT / "data/unified_bank.json").read_text())
        row = json.loads((ROOT / "data/claude_reference.jsonl").read_text().splitlines()[0])
        self.write(self.project / ".claude/settings.json", {"env": {
            "ANTHROPIC_AUTH_TOKEN": "sk-TESTTOKEN1234567",
            "ANTHROPIC_BASE_URL": "https://relay.invalid",
            "ANTHROPIC_DEFAULT_OPUS_MODEL": "claude-opus-5-5",
        }})
        output = self.project / "result.json"
        cwd = os.getcwd()
        os.chdir(self.project)
        try:
            with patch.object(llm_checker, "load_bank", return_value=bank), \
                 patch.object(llm_checker, "codex_executable", side_effect=AssertionError("codex used")), \
                 patch.object(llm_checker, "run_claude_api", return_value=(row["text"], {"output_tokens": 9})) as run, \
                 patch("sys.stdout", new_callable=__import__("io").StringIO) as stdout:
                self.assertEqual(llm_checker.main(["-m", "Opus", "-n", "1", "--timeout", "30",
                                                   "--output", str(output)]), 0)
        finally:
            os.chdir(cwd)
        config, prompt, timeout = run.call_args.args
        self.assertEqual(config["model"], "claude-opus-5-5")
        self.assertEqual(config["url"], "https://relay.invalid/v1/messages")
        self.assertEqual(timeout, 30)
        self.assertIn("测试模型：opus → claude-opus-5-5", stdout.getvalue())
        self.assertIn("sk-T...4567", stdout.getvalue())
        self.assertNotIn("sk-TESTTOKEN1234567", stdout.getvalue())
        saved = json.loads(output.read_text())
        self.assertEqual(saved["api_model"], "claude-opus-5-5")


class ClaudeRequestTests(unittest.TestCase):
    CONFIG = {"model": "claude-opus-5-5", "url": "https://relay.invalid/v1/messages", "proxies": {},
              "headers": {"authorization": "Bearer sk-test", "anthropic-version": "2023-06-01",
                          "content-type": "application/json"}}

    def respond(self, payload):
        response = Mock()
        response.read.return_value = json.dumps(payload).encode()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        opener = Mock()
        opener.open.return_value = response
        return patch.object(llm_checker, "_claude_opener", return_value=opener), opener

    def test_request_is_bare_messages_call(self):
        patcher, opener = self.respond({
            "type": "message", "stop_reason": "end_turn", "usage": {"output_tokens": 12},
            "content": [{"type": "thinking", "thinking": "..."}, {"type": "text", "text": "1, 2, 3"}]})
        with patcher:
            answer, usage = llm_checker.run_claude_api(self.CONFIG, "prompt", timeout=10)
        self.assertEqual(answer, "1, 2, 3")
        self.assertEqual(usage["output_tokens"], 12)
        request = opener.open.call_args.args[0]
        self.assertEqual(request.full_url, "https://relay.invalid/v1/messages")
        self.assertEqual(request.get_header("Authorization"), "Bearer sk-test")
        body = json.loads(request.data)
        self.assertEqual(body["model"], "claude-opus-5-5")
        self.assertEqual(body["messages"], [{"role": "user", "content": "prompt"}])
        self.assertNotIn("system", body)
        self.assertNotIn("temperature", body)
        self.assertLessEqual(opener.open.call_args.kwargs["timeout"], 10)

    def test_refusal_and_truncation_are_rejected(self):
        for reason, message in (("refusal", "拒绝"), ("max_tokens", "截断")):
            patcher, _ = self.respond({"stop_reason": reason, "content": [{"type": "text", "text": "1"}]})
            with patcher, self.assertRaisesRegex(RuntimeError, message):
                llm_checker.run_claude_api(self.CONFIG, "prompt")

    def test_http_error_message_is_compacted(self):
        import io
        import urllib.error

        error = urllib.error.HTTPError(self.CONFIG["url"], 401, "Unauthorized", {},
                                       io.BytesIO(b'{"type":"error","error":{"message":"invalid x-api-key"}}'))
        opener = Mock()
        opener.open.side_effect = error
        with patch.object(llm_checker, "_claude_opener", return_value=opener):
            with self.assertRaisesRegex(RuntimeError, "HTTP 401：invalid x-api-key"):
                llm_checker.run_claude_api(self.CONFIG, "prompt")
        self.assertEqual(opener.open.call_count, 1)

    def test_timeout_is_reported(self):
        opener = Mock()
        opener.open.side_effect = TimeoutError("timed out")
        with patch.object(llm_checker, "_claude_opener", return_value=opener):
            with self.assertRaisesRegex(RuntimeError, "超过 5 秒未返回"):
                llm_checker.run_claude_api(self.CONFIG, "prompt", timeout=5)

if __name__ == "__main__":
    unittest.main()
