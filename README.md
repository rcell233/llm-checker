# llm-checker

用本地 Codex CLI，或者直接读取 Claude Code 的 API 配置，执行 [ModelTrace](https://github.com/xqy2006/ModelTrace) 的长整数指纹探针，并在终端显示候选模型归因。

## 快速开始

前提：可用 `python3`。检测 Codex 需要已安装并登录 `codex` CLI；检测 Claude 只需要 Claude Code 已配置 API（不需要调用 `claude` 命令）。

### 一键测试

```sh
wget -qO- "https://raw.githubusercontent.com/rcell233/llm-checker/main/llm_checker.py" | python3 - -m gpt-6-astra
```

或在本地仓库运行：

```sh
python3 llm_checker.py -m gpt-6-astra
```

### 参数说明

- `-m, --model`：codex 模型名（省略则使用 Codex 默认模型）；`fable`、`opus`、`sonnet`、`haiku` 会切换到 Claude Code API 检测（见下文）
- `-r, --reasoning`：Codex 推理等级 `low/medium/high/xhigh`（默认 `low`，Claude 档位不使用）
- `--timeout`：单道探针超时秒数（默认 120）
- `-n, --number`：探针数量（默认 3）
- `-j, --max-concurrency`：并发探针数（默认 3）
- `--list-models`：列出指纹库中的候选模型
- `--output <file>`：保存提示词、原始回答和评分结果为 JSON
- `--bank <path>`：使用自定义指纹库

运行脚本只使用 Python 标准库；首次会从本仓库下载约 834 KB 的统一指纹库。如果在本仓库目录运行，会直接读取本地 `data/unified_bank.json`。

当前指纹库同步自 [ModelTrace `55a2e4a`](https://github.com/xqy2006/ModelTrace/commit/55a2e4a55170423b484d701e9a82ab62b268c811)，共 16 个候选模型，包含 `gpt-6-sol`、`gpt-6-luna` 和 `claude-opus-5-5`。原始参考数据与统一指纹库均直接采用该上游版本。

### Claude Code（API 方式）

`-m` 为 `fable`、`opus`、`sonnet` 或 `haiku`（不区分大小写）时，脚本会读取 Claude Code 的配置，**绕过 `claude` CLI，直接发送一条不带系统提示词的 Anthropic Messages 请求**（`POST {ANTHROPIC_BASE_URL}/v1/messages`）。这样做是因为 Claude Code 的系统提示词会明显改变数字偏好，而指纹库是用裸 API 请求采集的。

```sh
cd /path/to/your/project   # 在项目目录运行，才会读到项目级 .claude/ 配置
python3 llm_checker.py -m opus
```

只支持用 API 方式使用的 Claude Code。官方账号（订阅 / OAuth）登录，以及 Bedrock、Vertex、Foundry 这类云厂商配置都会直接报错退出。

配置读取顺序与 Claude Code 保持一致，后面的覆盖前面的：

1. Shell 环境变量
2. 用户配置 `~/.claude/settings.json`（设置了 `CLAUDE_CONFIG_DIR` 时读取该目录）
3. 项目配置 `./.claude/settings.json`
4. 项目本地配置 `./.claude/settings.local.json`
5. 托管配置：`managed-settings.json` 及 `managed-settings.d/*.json`（Linux/WSL 为 `/etc/claude-code/`，macOS 为 `/Library/Application Support/ClaudeCode/`，Windows 为 `C:\Program Files\ClaudeCode\`）

settings 文件 `env` 中的变量会覆盖同名 Shell 环境变量；值为空字符串视为未设置。用到的配置项：

| 配置项 | 用途 |
| --- | --- |
| `ANTHROPIC_BASE_URL` | API 端点，默认 `https://api.anthropic.com` |
| `ANTHROPIC_AUTH_TOKEN` | 以 `Authorization: Bearer` 发送（优先） |
| `ANTHROPIC_API_KEY` | 以 `x-api-key` 发送 |
| `apiKeyHelper`（settings 顶层） | 执行该命令，输出同时作为 `x-api-key` 和 Bearer 发送 |
| `ANTHROPIC_DEFAULT_{FABLE,OPUS,SONNET,HAIKU}_MODEL` | 档位对应的模型 ID；末尾的 `[1m]` 会被去掉 |
| `ANTHROPIC_DEFAULT_*_MODEL_NAME` | 没有 `*_MODEL` 时的后备（官方定义为 `/model` 菜单中的显示名，仅在值形如模型 ID 时采用） |
| `ANTHROPIC_CUSTOM_HEADERS`、`ANTHROPIC_BETAS` | 附加请求头（每行一个 `Name: Value`）和 `anthropic-beta` |
| `HTTPS_PROXY`、`HTTP_PROXY`、`NO_PROXY` | 代理设置 |

档位没有配置模型时，使用内置默认值：`fable` → `claude-fable-5-1`，`opus` → `claude-opus-5-5`，`sonnet` → `claude-sonnet-5-5`，`haiku` → `claude-haiku-4-5-20251001`。运行时会打印实际使用的端点、脱敏后的凭证、模型 ID，以及每一项分别来自哪个文件。

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
| `llm_checker.py` | 纯标准库的测试入口（Codex CLI / Claude Code API） |
| `data/unified_bank.json` | 模型中心、环境方向和校准参数 |
| `data/*_reference.jsonl` | 原始回答与模型标签 |
| `challenge_suite.py` | 36 条挑战及环境设置 |
| `fingerprint.py`、`bank_builder.py` | 特征提取与拟合逻辑 |

## 致谢

- [ModelTrace](https://github.com/xqy2006/ModelTrace) - 原始算法和参考数据来源
