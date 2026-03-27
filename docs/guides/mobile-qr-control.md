# Mobile QR Control

This page follows the README's mobile control flow through `civ6_tacticall`.

## What It Is

`civ6_tacticall` is the separate mobile controller and relay project used to drive CivStation from a phone browser through a QR-based pairing flow.

## Minimal Setup

```bash
git clone https://github.com/minsing-jin/civ6_tacticall.git
cd civ6_tacticall
npm install
npm start
```

Then prepare the host bridge config:

```bash
cp host-config.example.json host-config.json
```

Important values:

- `relayUrl`: `ws://127.0.0.1:8787/ws`
- `localApiBaseUrl`: `http://127.0.0.1:8765`
- `localAgentUrl`: `ws://127.0.0.1:8765/ws`

## Start the Bridge

```bash
npm run host
```

The bridge:

1. connects to the mobile relay
2. connects to the local CivStation runtime
3. prints a QR code for pairing

## Why This Flow Exists

The point is not "mobile for the sake of mobile."

The point is to keep the operator in the loop without covering the game window or tying control to the same display the agent is acting on.
