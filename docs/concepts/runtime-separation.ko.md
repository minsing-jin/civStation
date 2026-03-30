# 런타임 분리

README는 CivStation을 하나의 루프가 모든 책임을 다 하는 시스템으로 설명하지 않습니다. 서로 다른 종류의 일이 서로 다른 런타임 lane에서 돌아가도록 설명합니다.

## 세 가지 런타임 lane

### Background runtime

action loop 옆에서 돌아가는 것이 유리한 일들:

- context observation
- turn tracking
- strategy refresh
- background reasoning

### Main-thread action runtime

결정적이고 interruptible해야 하는 lane:

- 현재 화면 routing
- primitive action planning
- 실제 game window 위 action execution

### HitL runtime

제어를 action thread 밖에 두는 lane:

- dashboard
- WebSocket control
- relay와 mobile controller
- strategy와 lifecycle directives

## 왜 중요한가

이걸 모두 하나의 blocking loop에 섞어 넣으면:

- 비싼 reasoning이 action execution을 멈추고
- interrupt가 취약해지고
- strategy refinement가 늦게 도착하고
- external control은 부가 기능처럼 밀려납니다

런타임 분리가 있어야 CivStation이 단순히 영리한 데모가 아니라 실제로 운영 가능한 시스템이 됩니다.
