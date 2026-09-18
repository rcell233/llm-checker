# llm-checker

用本地 Codex CLI 执行 [ModelTrace](https://github.com/xqy2006/ModelTrace) 的长整数指纹探针，识别 LLM 模型归属。

## 用法

前提：已安装并登录 `codex` CLI，且可用 `python3`。

```bash
python3 llm_checker.py -m gpt-6-astra
```

### 一键运行

以下任选其一：

```bash
wget -qO- "https://raw.githubusercontent.com/rcell233/llm-checker/main/llm_checker.py" | python3 - -m gpt-6-astra
```

```bash
curl -fsSL "https://raw.githubusercontent.com/rcell233/llm-checker/main/llm_checker.py" | python3 - -m gpt-6-astra
```

参数：

- `-m, --model`：codex 模型名，省略则用本地默认
- `-r, --reasoning`：推理等级 `low/medium/high/xhigh`（默认 `low`）
- `-n, --number`：探针数量（默认 3）
- `-j, --max-concurrency`：并发数（默认 1）
- `--list-models`：列出指纹库中的候选模型
- `--output`：保存结果为 JSON
- `--bank`：使用自定义指纹库

结果是**当前候选模型集合内**的相对概率，不能证明实际模型身份。

## 扩展新模型

安装依赖：

```bash
pip install -r requirements.txt
```

添加新模型（支持 OpenAI 和 Anthropic API）：

```bash
export OPENAI_API_KEY='...'
python3 add_model.py --label gemini-example --family gemini \
  --api-model gemini-example --base-url https://your-provider.example/v1
```

完成后自动重建 `data/unified_bank.json`。更多选项参见 `python3 add_model.py --help`。

## 致谢

- [ModelTrace](https://github.com/xqy2006/ModelTrace) - 原始算法和数据来源
