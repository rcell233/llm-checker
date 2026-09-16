import json
import unittest
from pathlib import Path
from unittest.mock import patch

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
             patch.object(llm_checker, "run_codex", return_value=(row["text"], {"output_tokens": 10})):
            self.assertEqual(llm_checker.main(["-m", "gpt-5.5", "-n", "1"]), 0)


if __name__ == "__main__":
    unittest.main()
