# 安装

先从可以在本地真正跑起来的最短路径开始。

## 前置条件

- Python `3.10+`
- 一个代理可以观察和控制的本地 Civilization VI 环境
- 你计划使用的模型 provider 的 API Key
- 操作系统中的屏幕录制与输入控制权限

如果你要在 Linux 上启用语音或完整 dashboard 栈，可能还需要 PortAudio 等系统包。CI 也因此会在 Ubuntu 上安装 `portaudio19-dev` 和 `gcc`。

## 基础安装

日常开发使用：

```bash
make install
```

这会以 editable 模式安装项目，并安装测试依赖与 `pre-commit`。

## 安装 docs 支持

如果你要在本地预览或构建文档站点：

```bash
uv pip install -e ".[docs,test]"
```

之后可以直接使用：

```bash
make docs-serve
make docs-build
```

## 环境变量

只设置你会实际使用的 provider 即可。

```env
ANTHROPIC_API_KEY=...
GENAI_API_KEY=...
OPENAI_API_KEY=...
RELAY_TOKEN=...
```

provider 别名见 [提供商与图像管线](../guides/providers-and-image-pipeline.md)。

## 验证安装

检查 CLI：

```bash
python -m civStation.agent.turn_runner --help
```

检查 MCP server 入口：

```bash
python -m civStation.mcp.server
```

MCP server 默认使用 stdio transport，因此它会等待客户端，而不是打开浏览器。

## 下一步

继续阅读 [快速开始](quickstart.md)。
