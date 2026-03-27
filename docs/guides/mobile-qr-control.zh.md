# 移动端 QR 控制

这一页直接沿用 README 中通过 `civ6_tacticall` 进行移动端控制的流程。

## 这是什么

`civ6_tacticall` 是独立的 mobile controller / relay 项目，用 QR 配对的方式让你通过手机浏览器控制 CivStation。

## 最小设置

```bash
git clone https://github.com/minsing-jin/civ6_tacticall.git
cd civ6_tacticall
npm install
npm start
```

然后准备 host bridge config：

```bash
cp host-config.example.json host-config.json
```

关键值：

- `relayUrl`: `ws://127.0.0.1:8787/ws`
- `localApiBaseUrl`: `http://127.0.0.1:8765`
- `localAgentUrl`: `ws://127.0.0.1:8765/ws`

## 启动 bridge

```bash
npm run host
```

bridge 会：

1. 连接移动端 relay
2. 连接本地 CivStation runtime
3. 打印用于配对的 QR code

## 为什么需要这条流程

重点不是“为了移动端而移动端”。

重点是让操作员保持在 loop 中，同时又不遮挡代理正在操作的游戏画面。
