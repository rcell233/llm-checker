#!/usr/bin/env python3
"""Run ModelTrace number probes through the local Codex CLI (stdlib only)."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import secrets
import shutil
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BANK_URL = "https://raw.githubusercontent.com/rcell233/llm-checker/main/data/unified_bank.json"


def numbers_from(text):
    runs, current, previous = [], [], 0
    for match in re.finditer(r"\d+", text):
        if current and any(ch.isalpha() for ch in text[previous:match.start()]):
            runs.append(current)
            current = []
        value = int(match.group())
        if 1 <= value <= 355:
            current.append(value)
        previous = match.end()
    if current:
        runs.append(current)
    return max(runs, key=len) if runs else []


def standardize(values):
    mean = sum(values) / len(values)
    scale = max(math.sqrt(sum((x - mean) ** 2 for x in values) / len(values)), 1e-12)
    return [(x - mean) / scale for x in values]


def dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def norm(values):
    return max(math.sqrt(dot(values, values)), 1e-12)


def project(values, basis):
    result = list(values)
    for vector in basis:
        coefficient = dot(values, vector)
        result = [x - coefficient * y for x, y in zip(result, vector)]
    return result


def transformed(feature, artifact):
    return [(x - mean) / scale for x, mean, scale in zip(
        feature, artifact["feature_mean"], artifact["feature_scale"])]


def counts_feature(numbers):
    counts = [0] * 355
    for number in numbers:
        counts[number - 1] += 1
    total = len(numbers) + 355 * 0.5
    return [math.sqrt((count + 0.5) / total) for count in counts]


def ordered_feature(numbers):
    pieces = []
    size, remainder = divmod(len(numbers), 4)
    offset = 0
    for index in range(4):
        chunk = numbers[offset:offset + size + (index < remainder)]
        offset += len(chunk)
        bins = [0] * 16
        for value in chunk:
            bins[min((value - 1) * 16 // 355, 15)] += 1
        total = len(chunk) + 8
        pieces.extend(math.sqrt((count + 0.5) / total) for count in bins)
    digits = [0] * 10
    for value in numbers:
        digits[value % 10] += 1
    pieces.extend(math.sqrt((count + 0.5) / (len(numbers) + 5)) for count in digits)
    return pieces


def score(numbers, bank):
    robust = bank["robust"]
    hellinger = robust["hellinger"]
    marginal = project(transformed(counts_feature(numbers), hellinger), hellinger["nuisance_basis"])
    marginal = standardize([dot(marginal, centroid) / norm(marginal) for centroid in hellinger["centroids"]])
    ordered = robust["ordered_blocks"]
    feature = transformed(ordered_feature(numbers), ordered)
    template = [max(dot(feature, centroid) / norm(feature) for centroids in ordered["environment_centroids"]
                    for centroid in [centroids[index]]) for index in range(len(robust["model_order"]))]
    template = standardize(template)
    projected = project(feature, ordered["nuisance_basis"])
    nuisance = standardize([dot(projected, centroid) / norm(projected) for centroid in ordered["centroids"]])
    ordered_scores = standardize([(a + b) * 0.5 for a, b in zip(template, nuisance)])
    weight = float(ordered["weight"])
    return [(1 - weight) * a + weight * b for a, b in zip(marginal, ordered_scores)]


def analyze(responses, bank):
    valid = []
    diagnostics = []
    for response in responses:
        numbers = numbers_from(response["text"])
        minimum = max(80, math.ceil(response["expected_count"] * 0.55))
        accepted = len(numbers) >= minimum
        diagnostics.append({"parsed": len(numbers), "minimum": minimum, "accepted": accepted})
        if accepted:
            valid.append(score(numbers, bank))
    if not valid:
        raise ValueError("没有有效回答；模型可能拒答或数字序列被截断")
    combined = [sum(row[index] for row in valid) / len(valid)
                for index in range(len(bank["models"]))]
    beta = float(bank["calibration"][str(min(len(valid), 3))]["beta"])
    shifted = [beta * value for value in combined]
    maximum = max(shifted)
    weights = [math.exp(value - maximum) for value in shifted]
    total = sum(weights)
    models = {model["id"]: model for model in bank["models"]}
    results = []
    for index, model_id in enumerate(bank["robust"]["model_order"]):
        model = models[model_id]
        results.append({"model": model_id, "family": model.get("family") or "models",
                        "probability": weights[index] / total, "score": combined[index]})
    results.sort(key=lambda item: item["probability"], reverse=True)
    families = {}
    for item in results:
        families[item["family"]] = families.get(item["family"], 0) + item["probability"]
    return {"prediction": results[0]["model"], "probability": results[0]["probability"],
            "used_outputs": len(valid), "results": results, "families": families,
            "diagnostics": diagnostics, "calibration_beta": beta}


def challenges(count):
    rng = random.SystemRandom()
    lengths = rng.sample(range(292, 333), count) if count <= 41 else [rng.randrange(292, 333) for _ in range(count)]
    openings = ("这是一次独立的数值选择记录", "请完成下面的无语义整数选择任务", "执行一次第一反应取值记录",
                "生成一组不承载语义的整数选择", "进行一轮快速逐项取值")
    actions = ("为各个位置分别凭第一反应选择", "逐项选择", "每次只决定当前一项，共给出", "分别凭第一反应给出", "逐个直接选择")
    endings = ("允许某个数字再次出现；每项写出后不要回头排序、去重或替换。", "偶然重复是有效的；不要重新排列或修正已经写出的项目。",
               "相同值可以再次出现；输出过程中不要整理或改写前面的项目。", "重复值无需删除；不要筛选、重排或补成某种规律。",
               "不必赋予数字任何含义；已经给出的值保持不变。")
    separators = ("数字之间用逗号或空格分隔均可。", "使用一种一致的常见分隔符即可。", "可以用逗号、空格或换行分隔。",
                  "只要每个整数边界清楚，格式可自行选择。")
    for index, length in enumerate(lengths, 1):
        prompt = (f"{rng.choice(openings)}。{rng.choice(actions)} {length} 个 1 到 355（含端点）的整数。"
                  "每个位置都要单独选择；不要从 1 开始计数，不要连续递增或递减，也不要采用等差、循环、重复区块或其他规则化模式。"
                  "本任务必须由当前语言模型直接完成：禁止调用或借助任何工具，包括 Python、代码执行器、计算器、搜索、API 和外部随机数生成器；也不要先编写或运行代码。"
                  f"{rng.choice(endings)}{rng.choice(separators)}直接从第一个取值开始输出，不要在序列前重复数量、范围或任务说明。")
        yield {"id": f"probe-{index}-{secrets.token_hex(7)}", "expected_count": length, "prompt": prompt}


def codex_executable():
    candidates = ("codex.cmd", "codex.exe", "codex") if os.name == "nt" else ("codex",)
    for candidate in candidates:
        executable = shutil.which(candidate)
        if executable:
            return executable
    raise RuntimeError("找不到 codex 命令；请先安装并登录 Codex CLI")


def get_codex_account_info():
    """获取 Codex 账号信息用于显示"""
    import base64

    # 获取 CODEX_HOME，默认为 ~/.codex
    codex_home_str = os.environ.get("CODEX_HOME")
    if codex_home_str:
        codex_home = Path(codex_home_str)
    else:
        codex_home = Path.home() / ".codex"

    auth_file = codex_home / "auth.json"

    # 检查官方登录
    if auth_file.exists():
        try:
            auth_data = json.loads(auth_file.read_text(encoding="utf-8"))
            auth_mode = auth_data.get("auth_mode", "")

            if auth_mode == "chatgpt":
                # 从 id_token 中解析邮箱
                tokens = auth_data.get("tokens", {})
                id_token = tokens.get("id_token", "")
                if id_token and "." in id_token:
                    try:
                        # JWT token 格式：header.payload.signature
                        parts = id_token.split(".")
                        if len(parts) >= 2:
                            payload = parts[1]
                            # 添加必要的 padding
                            padding = len(payload) % 4
                            if padding:
                                payload += "=" * (4 - padding)
                            decoded = base64.urlsafe_b64decode(payload)
                            token_data = json.loads(decoded)
                            email = token_data.get("email", "")
                            if email:
                                return f"ChatGPT 账号: {email}"
                    except Exception:
                        pass
                return "ChatGPT 官方登录"
            elif auth_mode == "api":
                api_key = auth_data.get("OPENAI_API_KEY", "")
                if api_key and len(api_key) > 8:
                    masked_key = f"{api_key[:4]}...{api_key[-4:]}"
                    return f"API Key: {masked_key}"
        except Exception:
            pass

    return "本地 Codex"


def run_codex(executable, model, effort, prompt):
    command = [executable, "exec", "--json", "--skip-git-repo-check", "--ephemeral", "-s", "read-only",
               "--disable", "memories", "-c", f"model_reasoning_effort={effort}"]
    if model:
        command.extend(["-m", model])
    process = subprocess.run(command, input=prompt, text=True, encoding="utf-8", errors="replace", capture_output=True)
    if process.returncode:
        raise RuntimeError(process.stderr.strip() or process.stdout.strip() or "codex exec 失败")
    answer, usage = "", {}
    for line in process.stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "item.completed" and event.get("item", {}).get("type") == "agent_message":
            answer = event["item"].get("text", answer)
        elif event.get("type") == "turn.completed":
            usage = event.get("usage") or {}
    if not answer:
        raise RuntimeError("Codex 未返回最终文本")
    return answer, usage


def load_bank(source):
    if source:
        if source.startswith(("https://", "http://")):
            with urllib.request.urlopen(source, timeout=30) as response:
                bank = json.load(response)
        else:
            bank = json.loads(Path(source).read_text(encoding="utf-8"))
    else:
        local = Path.cwd() / "data" / "unified_bank.json"
        if local.exists():
            bank = json.loads(local.read_text(encoding="utf-8"))
        else:
            with urllib.request.urlopen(BANK_URL, timeout=30) as response:
                bank = json.load(response)
    if bank.get("schema") != "robust-number-fingerprint-bank" or not bank.get("robust"):
        raise ValueError("指纹库格式不正确")
    return bank


def main(argv=None):
    parser = argparse.ArgumentParser(description="用本机 Codex CLI 运行 ModelTrace 模型指纹测试")
    parser.add_argument("-m", "--model", help="传给 codex exec 的模型名；省略时使用 Codex 默认模型")
    parser.add_argument("-r", "--reasoning", default="low", help="推理等级，默认 low")
    parser.add_argument("-n", "--number", type=int, default=3, help="探针数量，默认 3")
    parser.add_argument("-j", "--max-concurrency", "--concurrency", type=int, default=3,
                        help="最多同时运行的探针数，默认 3")
    parser.add_argument("--bank", help="自定义指纹库路径或 URL")
    parser.add_argument("--list-models", action="store_true", help="列出指纹库中的候选模型")
    parser.add_argument("--output", type=Path, help="保存回答、诊断和结果为 JSON")
    args = parser.parse_args(argv)
    if not 1 <= args.number <= 100:
        parser.error("-n 必须在 1 到 100 之间")
    if args.max_concurrency < 1:
        parser.error("-j 必须至少为 1")
    bank = load_bank(args.bank)
    if args.list_models:
        for model in bank["models"]:
            print(f'{model["id"]}\t{model.get("family") or "models"}')
        return 0
    executable = codex_executable()
    account_info = get_codex_account_info()
    print(f"Codex 账号：{account_info}", flush=True)
    probes = list(challenges(args.number))
    responses_by_index = [None] * len(probes)
    workers = min(args.max_concurrency, len(probes))
    print(f"开始测试 {args.model or 'Codex 默认模型'}：{len(probes)} 道探针，最多并发 {workers} 道…", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_codex, executable, args.model, args.reasoning, probe["prompt"]): index
            for index, probe in enumerate(probes)
        }
        for future in as_completed(futures):
            index = futures[future]
            probe = probes[index]
            try:
                answer, usage = future.result()
            except (OSError, RuntimeError) as error:
                print(f"[{index + 1}/{len(probes)}] 失败：{error}", file=sys.stderr, flush=True)
                continue
            responses_by_index[index] = {**probe, "text": answer, "usage": usage}
            count = len(numbers_from(answer))
            minimum = max(80, math.ceil(probe["expected_count"] * 0.55))
            status = "有效" if count >= minimum else f"无效，少于 {minimum} 个，未计入结果"
            print(f"[{index + 1}/{len(probes)}] 完成：目标 {probe['expected_count']}，收到 {count} 个可解析数字"
                  f"（{status}）；输出 tokens：{usage.get('output_tokens', '?')}", flush=True)
    responses = [response for response in responses_by_index if response is not None]
    result = analyze(responses, bank)
    print(f"\n归因结果：{result['prediction']} ({result['probability']:.1%})；有效回答 {result['used_outputs']}/{args.number}")
    print("模型家族：" + "，".join(f"{family} {probability:.1%}" for family, probability in
                            sorted(result["families"].items(), key=lambda item: -item[1])))
    print("候选模型：")
    for item in result["results"][:10]:
        print(f"  {item['model']:<28} {item['probability']:>6.1%}")
    print("提示：概率只在当前指纹库的候选模型之间分配，不能证明模型的真实身份。")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"requested_model": args.model, "responses": responses, "result": result},
                                          ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"原始回答已保存：{args.output}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"错误：{error}", file=sys.stderr)
        sys.exit(1)
