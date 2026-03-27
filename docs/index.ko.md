<div class="hero-grid" markdown>

<div class="hero-panel" markdown>

<span class="hero-kicker">개요</span>

<div class="language-switch" markdown>

<a class="md-button" href="../">English</a>
<a class="md-button md-button--primary" href="./">한국어</a>
<a class="md-button" href="../zh/">中文</a>

</div>

<h1 class="landing-title">CivStation</h1>

비전-언어 모델이 Civilization VI를 플레이할 수 있게 만드는 런타임과 툴링입니다.

<p class="landing-lead">CivStation은 VLM을 Civilization VI UI에 실제로 연결하기 위한 프로젝트입니다. 화면 관찰, primitive routing, action planning, local execution, evaluation, human override를 한 시스템 안에서 다룹니다.</p>

[설치하고 실행하기](getting-started/installation.md){ .md-button .md-button--primary }
[라이브 세션 실행](getting-started/first-live-run.md){ .md-button }
[아키텍처 읽기](concepts/mental-model.md){ .md-button }

</div>

</div>

<div class="ornament-rule"></div>

## 여기서 시작하세요

<div class="grid cards" markdown>

-   **빠른 시작**

    ---

    프로젝트 설치, API 키 설정, 상태 UI와 함께 실행, 대시보드 열기까지 가장 짧은 경로입니다.

    [빠른 시작으로 이동](getting-started/quickstart.md)

-   **모바일 QR 제어**

    ---

    README의 `civ6_tacticall` 흐름을 따라, 운영자가 게임 화면을 덮지 않고 휴대폰에서 개입하는 방식을 설명합니다.

    [모바일 제어 가이드 열기](guides/mobile-qr-control.md)

-   **왜 CivStation인가**

    ---

    구현 세부사항보다 먼저, 프로젝트가 왜 이런 구조와 철학을 택했는지 설명합니다.

    [철학 읽기](concepts/why-civstation.md)

-   **아키텍처 개요**

    ---

    VLM 런타임, HitL 전략 고도화, routing, planning, execution이 어떻게 맞물리는지 설명합니다.

    [개념 읽기](concepts/mental-model.md)

</div>

## CivStation이 실제로 다루는 것

<div class="grid cards" markdown>

-   **라이브 VLM 플레이**

    ---

    Civ6 화면을 VLM이 읽고, 현재 UI 상태를 primitive로 라우팅하고, 다음 행동을 계획해 로컬에서 실행합니다.

-   **운영자 제어**

    ---

    실행 중에도 pause, resume, stop, strategy change, primitive override, discussion이 가능합니다.

-   **Layered MCP**

    ---

    내부 import에 묶이지 않도록 sessions, prompts, resources, adapter overrides, workflow tools를 MCP 표면으로 제공합니다.

-   **평가 프레임워크**

    ---

    라이브 실행만이 아니라 bbox 기반 action evaluation과 관련 픽스처로 routing/action 품질을 따로 점검할 수 있습니다.

</div>

<div class="ornament-rule"></div>

## 문서 구성

| 섹션 | 제공 내용 |
| --- | --- |
| `Getting Started` | 설치, 빠른 시작, 첫 실행, 모바일 QR 제어 |
| `Concepts` | 왜 CivStation인가, 런타임 분리, 아키텍처 개요, 레이어, 실행 루프, HITL, MCP |
| `Guides` | 에이전트 운영, 제어 표면, providers, evaluation |
| `Reference` | CLI 플래그, `config.yaml`, MCP 도구 목록, 폴더 맵 |
| `Development` | 기여 흐름, 테스트, 확장 포인트, 릴리스 노트 |
| `Appendix` | 레거시 페이지, 역사적 요약, 설계 노트 |

## 정식 이름

문서, 스크립트, 통합 시 다음 이름을 일관되게 사용하세요.

- 제품명: `CivStation`
- GitHub 저장소: `minsing-jin/civStation`
- Python package: `civStation`
- Python module: `civStation`

## 추천 읽기 순서

1. [빠른 시작](getting-started/quickstart.md)부터 읽습니다.
2. [왜 CivStation인가](concepts/why-civstation.md), [런타임 분리](concepts/runtime-separation.md), [아키텍처 개요](concepts/mental-model.md)를 읽습니다.
3. 일상 운영에는 [모바일 QR 제어](guides/mobile-qr-control.md), [에이전트 실행](guides/running-the-agent.md), [제어와 디스커션](guides/control-and-discussion.md)을 봅니다.
4. 통합과 확장에는 [MCP 도구](reference/mcp-tools.md)와 [스택 확장](development/extending-the-stack.md)을 봅니다.
