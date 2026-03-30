# Historical Implementation Summary

This page captures the older implementation summary so it remains accessible from the docs portal.

## What That Summary Covered

The original implementation summary focused on an earlier phase of the project:

- pytest migration for static evaluator tests
- VLM provider integration
- primitive-level provider support
- prompt module organization
- evaluator-facing documentation

## Milestones From That Snapshot

### Test migration

The summary recorded a migration of evaluator tests to pytest with tolerance, parsing, and integration coverage.

### Provider integration

It documented the addition of:

- Claude support
- Gemini support
- GPT support
- provider factory helpers

### Primitive integration

It also captured an earlier stage where multiple primitives were wired to optional VLM providers and custom prompts.

## Why Keep It

This document is useful as project archaeology. It shows how the evaluator and provider stack matured before the current layered runtime and MCP surface took shape.

## Current Canonical Docs

For up-to-date usage, prefer:

- [Providers and Image Pipeline](../guides/providers-and-image-pipeline.md)
- [Evaluation](../guides/evaluation.md)
- [Project Layout](../reference/project-layout.md)
