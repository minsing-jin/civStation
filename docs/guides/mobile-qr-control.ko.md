# 모바일 QR 제어

이 페이지는 README의 `civ6_tacticall` 모바일 제어 흐름을 따릅니다.

## 이것이 무엇인가

`civ6_tacticall`은 QR 기반 페어링을 통해 휴대폰 브라우저에서 CivStation을 제어하게 해 주는 별도 mobile controller / relay 프로젝트입니다.

## 최소 설정

```bash
git clone https://github.com/minsing-jin/civ6_tacticall.git
cd civ6_tacticall
npm install
npm start
```

그 다음 host bridge config 준비:

```bash
cp host-config.example.json host-config.json
```

중요한 값:

- `relayUrl`: `ws://127.0.0.1:8787/ws`
- `localApiBaseUrl`: `http://127.0.0.1:8765`
- `localAgentUrl`: `ws://127.0.0.1:8765/ws`

## bridge 실행

```bash
npm run host
```

bridge는:

1. 모바일 relay에 연결하고
2. 로컬 CivStation runtime에 연결하고
3. pairing용 QR 코드를 출력합니다

## 왜 이 흐름이 필요한가

핵심은 "모바일이라서 멋있다"가 아닙니다.

에이전트가 행동하는 같은 화면을 controller가 덮지 않으면서, 사람이 계속 루프 안에 머물 수 있게 하려는 것입니다.
