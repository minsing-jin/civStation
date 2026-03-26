# Contributing

This page is the practical version of `CONTRIBUTING.md`.

## Setup

```bash
make install
```

If you are editing docs too:

```bash
uv pip install -e ".[docs,test]"
```

## Day-to-Day Commands

```bash
make lint
make format
make check
make test
make coverage
make docs-build
```

Or with `just`:

```bash
just qa
just docs-build
```

## Contribution Expectations

- keep changes small and reversible
- update docs when behavior changes
- add or update tests when logic changes
- prefer extending an existing layer cleanly over bypassing the architecture

## Good PR Scope

Good examples:

- a primitive improvement with matching tests
- a new MCP adapter or tool improvement
- a HitL surface fix with dashboard or API validation
- documentation that matches the current code instead of aspirational design

## Where to File Things

- bugs and regressions: GitHub issues
- feature proposals: GitHub issues or PRs with clear scope
- documentation gaps: PRs are usually the fastest path

## Local Docs Workflow

```bash
make docs-serve
```

That starts the MkDocs site locally so changes can be reviewed in a browser before merging.
