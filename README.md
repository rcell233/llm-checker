# llm-checker

用本机 Codex CLI 执行 [ModelTrace](https://github.com/xqy2006/ModelTrace) 的长整数指纹探针，并在终端显示候选模型归因。运行脚本只使用 Python 标准库；首次从本仓库下载约 670 KB 的统一指纹库。如果在本仓库目录运行，会直接读取本地 `data/unified_bank.json`。

## 一键测试

前提：已安装并登录 `codex` CLI，且可用 `python3`。例如测试 Codex 中的 `gpt-6-astra`，运行默认的 3 道探针：

```sh
wget -qO- "https://raw.githubusercontent.com/rcell233/llm-checker/main/llm_checker.py" | python3 - -m gpt-6-astra
```

或在本地仓库运行：

```sh
python3 llm_checker.py -m gpt-6-astra
```

`-n` 是一次归因所用的独立探针数量，默认 3；三份回答合起来出一次结果。`-j`（`--max-concurrency`）控制同时运行的探针上限，默认 1，所以默认顺序执行；`-j 3` 可让默认三道题并发，`-n 5 -j 5` 可五道并发。每道探针都调用一次 `codex exec`，消耗对应模型额度。`-r` 默认 `low`，可显式指定 `high` 等等级。`-m` 省略时使用 Codex 的默认模型。`--list-models` 查看库内候选；`--output results/run.json` 保存提示词、原始回答和评分结果；`--bank /path/to/unified_bank.json` 可使用自定义指纹库。脚本中的 `codex exec` 采用只读沙盒、临时会话并关闭跨会话记忆。

结果是**当前候选模型集合内**的相对概率，不能证明实际模型身份。库外模型也会被归到最相近的库内模型。环境中的系统提示、模型版本、推理设置和提供商实现都可能改变数字偏好；GPT 参考数据来自官方 Codex，Claude 参考数据来自 OAIPro，跨环境解读尤其需要谨慎。原始 ModelTrace 的 [说明](https://github.com/xqy2006/ModelTrace#%E5%A3%B0%E6%98%8E) 也指出这些限制。

## 指纹如何生成

ModelTrace 对每个已知模型收集多个环境、多个提示下的长整数序列（整数范围 1–355），从每份回答提取两类特征：

1. 各数字出现频率，经平滑和 Hellinger 变换。
2. 序列分成四段后的 16 档直方图，以及末位数字分布，用来保留部分顺序信息。

建库时，所有模型在同一特征空间内拟合：先估计共享环境造成的偏移方向并投影去除，再求每个模型的中心。测试时两部分得分按 `0.75 / 0.25` 融合，多份有效回答取平均。最后通过交叉验证所得的温度系数做全局 softmax；模型家族概率是其成员模型概率的和。这里的纯标准库评分实现与 ModelTrace 的 NumPy 版逐项对齐。

| 文件 | 内容 | 用途 |
| --- | --- | --- |
| `data/gpt_reference.jsonl`、`data/claude_reference.jsonl` | 逐条原始回答、模型标签、挑战及环境元数据 | 新增模型与重建时的训练来源 |
| `data/unified_bank.json` | 模型中心、环境方向、统计量和概率校准参数 | 一键脚本实际读取的指纹库 |
| `challenge_suite.py` | 建库所用的 36 条挑战及环境设置 | 新模型采样 |
| `fingerprint.py`、`bank_builder.py` | 原版特征提取与拟合逻辑 | 重建和校验 |
| `llm_checker.py` | 无第三方依赖的 Codex 测试入口 | 一键运行 |

上述参考数据及核心算法来自 ModelTrace，按仓库中的 [MIT 许可证](LICENSE) 使用；`llm_checker.py` 是本项目的单文件运行入口。

## 扩展新模型

建库需安装 NumPy：

```sh
python3 -m pip install -r requirements.txt
```

支持 OpenAI Chat Completions 与 Anthropic Messages 兼容的 API。以一个 OpenAI 兼容服务为例，API Key 放在环境变量中，不会写入参考数据：

```sh
export OPENAI_API_KEY='...'
python3 add_model.py --label gemini-example --family gemini \
  --api-model gemini-example --base-url https://your-provider.example/v1
```

默认采集全部 36 条挑战，成功一条便追加一条 JSONL；中断后重新运行会跳过已成功的挑战。完成后自动重建 `data/unified_bank.json`。`--format anthropic` 切换 Anthropic Messages 格式，`--api-key-env NAME` 指定密钥变量，`-n 3` 等可先做小规模试采，`--no-rebuild` 可以只采集后再运行 `python3 rebuild_unified_bank.py`。为了保持跨模型环境校准，正式纳入候选库时建议采齐 36 条，且沿用相同挑战与采集条件。

要手动导入原始回答，也可按现有 JSONL 行结构追加到 `data/<family>_reference.jsonl`，其中 `source` 是唯一模型 ID，`condition_id` 和 `challenge_id` 应与 `challenge_suite.py` 对应，`strict_valid` 表示回答达到数字数目阈值；随后运行 `python3 rebuild_unified_bank.py`。更新指纹库后提交该 JSON 文件，远程一键命令即可使用新候选模型。若采集来源或提示环境与现有数据差异较大，重建后的概率需要重新验证。
