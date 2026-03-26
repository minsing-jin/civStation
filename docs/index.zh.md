<div class="hero-grid" markdown>

<div class="hero-panel" markdown>

<span class="hero-kicker">概览</span>

<div class="language-switch" markdown>

<a class="md-button" href="../">English</a>
<a class="md-button" href="../ko/">한국어</a>
<a class="md-button md-button--primary" href="./">中文</a>

</div>

<h1 class="landing-title">CivStation</h1>

让视觉语言模型真正去玩 Civilization VI 的运行时与工具集。

<p class="landing-lead">CivStation 用来把 VLM 真正接到 Civilization VI 的 UI 上。它把截图观察、primitive routing、action planning、local execution、evaluation 和 human override 放进同一个工作系统里。</p>

[安装并运行](getting-started/installation.md){ .md-button .md-button--primary }
[启动实时会话](getting-started/first-live-run.md){ .md-button }
[阅读架构](concepts/mental-model.md){ .md-button }

</div>

</div>

<div class="ornament-rule"></div>

## 从这里开始

<div class="grid cards" markdown>

-   **快速开始**

    ---

    最短路径：安装项目、设置 API Key、启动状态 UI，并打开仪表盘。

    [进入快速开始](getting-started/quickstart.md)

-   **第一次实机运行**

    ---

    说明运行流程、仪表盘暴露的能力，以及日志和运行产物的位置。

    [启动实时会话](getting-started/first-live-run.md)

-   **架构概览**

    ---

    解释 VLM 运行时、HitL strategy refinement、routing、planning 与 execution 如何拼在一起。

    [阅读概念部分](concepts/mental-model.md)

</div>

## CivStation 实际覆盖什么

<div class="grid cards" markdown>

-   **实时 VLM 游戏运行**

    ---

    让 VLM 读取 Civ6 屏幕，把当前 UI 状态路由到 primitive，规划下一步动作，并在本地执行。

-   **操作员控制**

    ---

    运行中也可以 pause、resume、stop、修改 strategy、覆盖 primitive、发起 discussion。

-   **Layered MCP**

    ---

    通过 sessions、prompts、resources、adapter overrides 与 workflow tools 暴露同一架构，而不是依赖内部 imports。

-   **评测**

    ---

    除了实时运行，还可以通过 bbox 风格评测与相关夹具单独检查 routing 与 action 质量。

</div>

<div class="ornament-rule"></div>

## 文档结构

| 部分 | 你会得到什么 |
| --- | --- |
| `Getting Started` | 安装、快速开始、第一次运行 |
| `Concepts` | 架构概览、分层、执行循环、HITL、MCP |
| `Guides` | 运行代理、控制界面、providers、evaluation |
| `Reference` | CLI 参数、`config.yaml`、MCP 工具列表、目录结构 |
| `Development` | 贡献流程、测试、扩展点、发布说明 |
| `Appendix` | 旧页面、历史总结、设计说明 |

## 规范名称

在写文档、写脚本或集成时，请统一使用这些名称。

- 产品名：`CivStation`
- GitHub 仓库：`minsing-jin/civStation`
- Python package：`civStation`
- Python module：`civStation`

## 推荐阅读顺序

1. 从 [快速开始](getting-started/quickstart.md) 开始。
2. 阅读 [架构概览](concepts/mental-model.md) 和 [分层](concepts/layers.md)。
3. 日常运行请看 [运行代理](guides/running-the-agent.md) 与 [控制与讨论](guides/control-and-discussion.md)。
4. 做集成和扩展时请看 [MCP 工具](reference/mcp-tools.md) 与 [扩展栈](development/extending-the-stack.md)。
