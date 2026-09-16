#!/usr/bin/env python3
"""Collect reference responses for an API model and rebuild the shared bank."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from challenge_suite import fingerprint_suite
from enrollment import make_row, request_completion
from rebuild_unified_bank import main as rebuild

DATA = Path(__file__).resolve().parent / "data"


def main(argv=None):
    parser = argparse.ArgumentParser(description="将新模型的 API 回答加入 ModelTrace 指纹库")
    parser.add_argument("--label", required=True, help="指纹库中显示的唯一模型 ID")
    parser.add_argument("--family", required=True, help="家族 ID，如 gpt、claude、gemini")
    parser.add_argument("--api-model", required=True, help="API 请求中的模型名")
    parser.add_argument("--base-url", required=True, help="API Base URL")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="保存 API Key 的环境变量名")
    parser.add_argument("--format", choices=("openai", "anthropic"), default="openai")
    parser.add_argument("--temperature", type=float, help="可选温度；默认由服务商决定")
    parser.add_argument("-n", "--sample-count", type=int, default=36, help="采集条数，默认完整的 36 条")
    parser.add_argument("--no-rebuild", action="store_true", help="只采集，稍后手动重建统一指纹库")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[a-z][a-z0-9_-]*", args.family):
        parser.error("--family 只能使用小写英文字母、数字、下划线和连字符，且须以字母开头")
    if not args.label.strip():
        parser.error("--label 不能为空")
    suite = fingerprint_suite()
    if not 3 <= args.sample_count <= len(suite):
        parser.error(f"-n 必须在 3 到 {len(suite)} 之间")
    key = os.getenv(args.api_key_env)
    if not key:
        parser.error(f"环境变量 {args.api_key_env} 未设置")
    path = DATA / f"{args.family}_reference.jsonl"
    existing = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line] if path.exists() else []
    all_ids = {row["source"] for file in DATA.glob("*_reference.jsonl") for row in
               (json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line)}
    if args.label in all_ids and not any(row["source"] == args.label for row in existing):
        parser.error("模型 ID 已存在于另一家族")
    done = {row["challenge_id"] for row in existing if row["source"] == args.label and row.get("strict_valid")}
    selected = [suite[int(index * len(suite) / args.sample_count)] for index in range(args.sample_count)]
    added = 0
    for index, task in enumerate(selected, 1):
        if task["challenge_id"] in done:
            print(f"[{index}/{len(selected)}] 跳过已采集的 {task['challenge_id']}", flush=True)
            continue
        prompt = task["prompt"]
        if task["user_prefix"]:
            prompt = task["user_prefix"] + "\n\nFinal task:\n" + prompt
        print(f"[{index}/{len(selected)}] 采集 {args.label} / {task['challenge_id']}…", flush=True)
        try:
            answer = request_completion(args.base_url, key, args.api_model, prompt,
                                        args.temperature, args.format, task["system"])
            row = make_row(args.label, answer, task["condition"], task["challenge_id"],
                           expected_count=task["expected_count"],
                           temperature=args.temperature if args.temperature is not None else "provider_default",
                           wrapper_transport=task["transport"], provider="api", prompt=prompt,
                           base_prompt=task["prompt"], system_prompt=task["system"],
                           user_prefix=task["user_prefix"])
            if not row["strict_valid"]:
                print(f"  无效：仅解析到 {row['parsed_count']} 个数字；下次运行会重试", file=sys.stderr)
                continue
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            added += 1
            print(f"  已保存 {row['parsed_count']} 个数字", flush=True)
        except Exception as error:
            print(f"  失败：{error}；下次运行会重试", file=sys.stderr)
    print(f"本次新增 {added} 条；参考数据：{path}")
    if not args.no_rebuild:
        rebuild()
    return 0


if __name__ == "__main__":
    sys.exit(main())
