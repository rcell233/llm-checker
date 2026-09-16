from __future__ import annotations

import json
from pathlib import Path

from bank_builder import build_bank, read_rows


PROJECT = Path(__file__).resolve().parent
OUTPUT = PROJECT / "data" / "unified_bank.json"
DATA = PROJECT / "data"


def main() -> None:
    rows = []
    sources = sorted(
        DATA.glob("*_reference.jsonl"),
        key=lambda path: (
            {"gpt": 0, "claude": 1}.get(path.name.removesuffix("_reference.jsonl"), 2),
            path.name,
        ),
    )
    if not sources:
        raise ValueError("data/ 中没有 *_reference.jsonl 参考数据")
    for path in sources:
        family_id = path.name.removesuffix("_reference.jsonl")
        family_name = family_id.upper() if family_id == "gpt" else family_id.title()
        rows.extend(
            {
                **row,
                "family_id": family_id,
                "family_name": family_name,
            }
            for row in read_rows(path)
        )
    bank = build_bank(rows)
    OUTPUT.write_text(
        json.dumps(bank, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "models": len(bank["models"]),
                "responses": sum(model["response_count"] for model in bank["models"]),
                "calibration": bank["calibration"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
