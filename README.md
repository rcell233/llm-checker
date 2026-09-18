# llm-checker

用本地 Codex CLI 执行 [ModelTrace](https://github.com/xqy2006/ModelTrace) 的长整数指纹探针，在终端显示候选模型归因。

## 快速开始

前提：已安装并登录 `codex` CLI，且可用 `python3`。

### 一键测试

```sh
wget -qO- "https://raw.githubusercontent.com/rcell233/llm-checker/main/llm_checker.py" | python3 - -m gpt-6-astra
```

或在本地仓库运行：

```sh
python3 llm_checker.py -m gpt-6-astra
```

### 参数说明

- `-m, --model`：codex 模型名（省略则使用 Codex 默认模型）
- `-r, --reasoning`：推理等级 `low/medium/high/xhigh`（默认 `low`）
- `-n, --number`：探针数量（默认 3）
- `-j, --max-concurrency`：并发探针数（默认 3）
- `--list-models`：列出指纹库中的候选模型
- `--output <file>`：保存提示词、原始回答和评分结果为 JSON
- `--bank <path>`：使用自定义指纹库

运行脚本只使用 Python 标准库；首次会从本仓库下载约 670 KB 的统一指纹库。如果在本仓库目录运行，会直接读取本地 `data/unified_bank.json`。

### 结果说明

结果是**当前候选模型集合内**的相对概率，不能证明实际模型身份。库外模型也会被归到最相近的库内模型。环境中的系统提示、模型版本、推理设置和提供商实现都可能改变数字偏好；跨环境解读需要谨慎。

## 扩展新模型

建库需安装 NumPy：

```sh
pip install -r requirements.txt
```

支持 OpenAI Chat Completions 与 Anthropic Messages 兼容的 API：

```sh
export OPENAI_API_KEY='...'
python3 add_model.py --label gemini-example --family gemini \
  --api-model gemini-example --base-url https://your-provider.example/v1
```

默认采集全部 36 条挑战，成功一条便追加一条 JSONL；中断后重新运行会跳过已成功的挑战。完成后自动重建 `data/unified_bank.json`。

其他选项：
- `--format anthropic`：切换到 Anthropic Messages 格式
- `--api-key-env NAME`：指定密钥环境变量名
- `-n 3`：先做小规模试采
- `--no-rebuild`：只采集不重建，之后手动运行 `python3 rebuild_unified_bank.py`

## 指纹原理

ModelTrace 对每个已知模型收集多个环境、多个提示下的长整数序列（1-355），提取数字频率和顺序特征，通过跨模型拟合去除环境偏移，为每个模型建立中心向量。测试时用交叉验证校准的温度系数做全局 softmax，多份回答取平均。

| 文件 | 说明 |
| --- | --- |
| `llm_checker.py` | 纯标准库的 Codex 测试入口 |
| `data/unified_bank.json` | 模型中心、环境方向和校准参数 |
| `data/*_reference.jsonl` | 原始回答与模型标签 |
| `challenge_suite.py` | 36 条挑战及环境设置 |
| `fingerprint.py`、`bank_builder.py` | 特征提取与拟合逻辑 |

## 致谢

- [ModelTrace](https://github.com/xqy2006/ModelTrace) - 原始算法和参考数据来源
