# CoreAgent

CoreAgent是基于[CoreCoder](https://github.com/he-yufeng/CoreCoder)用 Python 实现的轻量级命令行 AI 编程代理。它可以连接 OpenAI
兼容接口，在终端中与模型对话，并让模型读取、搜索、修改项目文件或执行命令。

## 功能

- 支持 OpenAI 兼容接口，包括 OpenAI、DeepSeek 和本地 Ollama 等服务
- 可选使用 LiteLLM，连接其支持的其他模型服务
- 流式输出模型回复，并显示模型发起的工具调用
- 提供文件读取、搜索、编辑、命令执行和子代理等内置工具
- 并发执行连续的只读安全工具调用，同时保持结果的原始调用顺序
- 自动压缩过长的会话上下文
- 支持保存、查看和恢复会话
- 支持交互模式和一次性任务模式

## 安装

推荐使用 [uv](https://docs.astral.sh/uv/) 管理依赖。进入项目目录后执行：

```bash
uv sync --locked
```

如果修改了 `pyproject.toml` 中的依赖，需要先更新锁文件，再同步环境：

```bash
uv lock
uv sync
```

`--locked` 要求 `uv.lock` 与 `pyproject.toml` 完全一致，因此锁文件过期时命令会直接
失败，而不会自动修改锁文件。

## 配置

CoreAgent 会读取当前目录中的 `.env`。如果当前目录没有 `.env`，它会从当前目录开始
向上查找，直到用户主目录。`.env` 不会覆盖已经存在的系统环境变量。

可以从示例文件开始配置：

```bash
cp .env.example .env
```

最小配置只需要提供一个 API Key：

```dotenv
COREAGENT_API_KEY=your-api-key
```

完整配置如下：

```dotenv
COREAGENT_API_KEY=
COREAGENT_BASE_URL=
COREAGENT_MODEL=gpt-5.6-luna
COREAGENT_MAX_TOKENS=4096
COREAGENT_TEMPERATURE=0
COREAGENT_MAX_CONTEXT=128000
COREAGENT_PROVIDER=openai
```

## 模型服务示例

### OpenAI

```bash
export OPENAI_API_KEY=your-api-key
uv run --locked coreagent
```

如需指定其他模型：

```bash
uv run --locked coreagent --model your-model-name
```

### DeepSeek

```bash
export DEEPSEEK_API_KEY=your-api-key
export COREAGENT_BASE_URL=https://api.deepseek.com
export COREAGENT_MODEL=deepseek-chat
uv run --locked coreagent
```

### Ollama

先在本机启动 Ollama 及所需模型，然后将 CoreAgent 指向它的 OpenAI 兼容接口：

```bash
export COREAGENT_API_KEY=ollama
export COREAGENT_BASE_URL=http://localhost:11434/v1
export COREAGENT_MODEL=qwen2.5-coder
uv run --locked coreagent
```

这里的 `COREAGENT_API_KEY` 是为了满足客户端参数要求，本地 Ollama 通常不会验证该值。

### LiteLLM

LiteLLM 是可选依赖，需要单独安装：

```bash
uv sync --locked --extra litellm
```

然后选择 `litellm` provider，并使用 LiteLLM 支持的模型名称：

```bash
export COREAGENT_PROVIDER=litellm
export COREAGENT_MODEL=anthropic/claude-3-haiku
export COREAGENT_API_KEY=your-api-key
uv run --locked --extra litellm coreagent
```

具体模型名称、鉴权变量和接口能力取决于对应服务。CoreAgent 不会替代服务商自身的
账号、模型权限或额度配置。

## 使用

### 交互模式

```bash
uv run --locked coreagent
```

输入任务后按 `Enter` 提交；使用 `Esc+Enter` 插入换行。输入 `quit`、`exit`、
`/quit` 或 `/exit` 退出。

### 一次性任务

使用 `-p` 或 `--prompt` 执行一轮任务，模型完成后程序自动退出：

```bash
uv run --locked coreagent -p "检查当前项目并运行测试"
```

### 恢复会话

在交互模式中执行 `/save` 会返回会话 ID。之后可以恢复该会话：

```bash
uv run --locked coreagent --resume SESSION_ID
```

恢复时默认使用会话中保存的模型；显式传入 `--model` 可以覆盖它。

## 命令行选项

```text
-m, --model MODEL       指定模型名称
--base-url URL          指定 OpenAI 兼容接口地址
--api-key KEY           指定 API Key
-p, --prompt PROMPT     执行一次性任务，不进入交互模式
-r, --resume ID         恢复已保存的会话
-v, --version           显示版本号
-h, --help              显示命令行帮助
```

也可以直接查看当前版本和完整帮助：

```bash
uv run --locked coreagent --version
uv run --locked coreagent --help
```

## 交互命令

| 命令 | 作用 |
| --- | --- |
| `/help` | 显示交互命令帮助 |
| `/reset` | 清空当前对话历史 |
| `/model` | 显示当前模型 |
| `/model <name>` | 在当前会话中切换模型 |
| `/tokens` | 显示累计 token 使用量；已知模型可能同时显示费用估算 |
| `/compact` | 手动尝试压缩当前会话上下文 |
| `/diff` | 列出本次进程中由文件写入和编辑工具修改过的文件 |
| `/save` | 将当前会话保存到磁盘 |
| `/sessions` | 列出最近保存的会话 |
| `quit` | 退出 CoreAgent |

会话保存在 `~/.coreagent/sessions/`，交互输入历史保存在
`~/.coreagent_history`。

## 内置工具

| 工具 | 作用 | 调度方式 |
| --- | --- | --- |
| `bash` | 执行 shell 命令，支持超时和超长输出截断 | 串行 |
| `read_file` | 按行读取文件，可指定起始行和行数 | 可并发 |
| `write_file` | 创建或覆盖文件，必要时创建父目录 | 串行 |
| `edit_file` | 对唯一匹配的原始文本执行精确替换，并返回 diff | 串行 |
| `glob` | 使用 glob 模式查找文件和目录 | 可并发 |
| `grep` | 使用正则表达式搜索文件内容 | 可并发 |
| `agent` | 创建具有独立对话上下文的子代理处理子任务 | 串行 |

工具是否执行由模型决定。`agent` 子代理可以使用父代理的其他工具，但不能递归创建
新的子代理。

### 工具并发规则

只有明确标记为并发安全的 `read_file`、`glob` 和 `grep` 才会并发执行。调度器按模型
返回的工具调用顺序工作：

- 连续出现的多个并发安全工具组成一个并发批次
- `bash`、写入、编辑、子代理、未知工具以及没有安全标记的新工具都会形成串行屏障
- 遇到串行屏障时，会先等待前面的安全批次全部结束，再执行该工具
- 最后一批安全工具会在循环结束后执行
- 工具结果始终按照模型原始调用顺序返回
- 单个安全工具直接执行，不会为它单独创建线程池

这套规则只描述工具调用的调度顺序，并不提供事务、文件锁或跨进程隔离。

## 上下文与会话

当估算的上下文逐渐接近 `COREAGENT_MAX_CONTEXT` 时，CoreAgent 会分阶段处理历史：

1. 截短较早且过长的工具输出，保留开头和结尾
2. 使用模型总结较早的对话，并保留近期消息
3. 接近限制时进一步折叠历史，仅保留摘要和最近内容

上述 token 数量是基于字符数的近似估算，并非模型服务商返回的精确计数。上下文压缩
也可能丢失部分细节，重要修改仍应以项目文件和版本控制记录为准。

## 开发与测试

安装开发依赖：

```bash
uv sync --locked --extra dev
```

运行完整测试：

```bash
uv run --locked --extra dev pytest -q tests
```

运行代码检查：

```bash
uv run --locked --extra dev ruff check .
```

当前主要目录结构：

```text
coreagent/
├── agent.py          # Agent 主循环与工具调度
├── cli.py            # 命令行入口和交互界面
├── config.py         # 默认配置和环境变量加载
├── context.py        # 上下文估算与压缩
├── llm.py            # OpenAI 兼容客户端和 LiteLLM 客户端
├── session.py        # 会话保存、恢复和列表
└── tools/            # 内置工具实现
tests/                # 自动化测试
docs/                 # 设计和修改方案
```

## 安全说明

CoreAgent 可以执行命令并修改文件。`bash` 工具会拦截部分明显的破坏性命令，并限制
执行时间及返回给模型的输出长度，但它不是操作系统级沙箱，也不能识别所有危险操作。

建议只在受版本控制的项目中使用，运行前确认当前工作区状态，并为 API Key、生产数据
和系统权限设置适当的隔离。对于重要修改，应检查 `/diff` 的文件列表以及实际的版本
控制差异后再提交。
