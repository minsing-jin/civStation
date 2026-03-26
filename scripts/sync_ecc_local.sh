#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ECC_SRC="${1:-${ECC_SRC:-/tmp/everything-claude-code}}"

CODEX_DIR="$REPO_ROOT/.codex"
CLAUDE_DIR="$REPO_ROOT/.claude"
AGENTS_DIR="$REPO_ROOT/.agents"

log() {
  printf '[ecc-local] %s\n' "$*"
}

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    printf 'Missing required directory: %s\n' "$path" >&2
    exit 1
  fi
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    printf 'Missing required file: %s\n' "$path" >&2
    exit 1
  fi
}

copy_tree() {
  local source_dir="$1"
  local target_dir="$2"

  rm -rf "$target_dir"
  mkdir -p "$(dirname "$target_dir")"
  cp -R "$source_dir" "$target_dir"
}

copy_file() {
  local source_file="$1"
  local target_file="$2"

  mkdir -p "$(dirname "$target_file")"
  cp "$source_file" "$target_file"
}

generate_prompt_file() {
  local source_file="$1"
  local output_file="$2"
  local command_name="$3"

  {
    printf '# ECC Command Prompt: /%s\n\n' "$command_name"
    printf 'Source: commands/%s.md from affaan-m/everything-claude-code\n\n' "$command_name"
    printf 'Use this prompt to run the ECC `%s` workflow in Codex.\n\n' "$command_name"
    awk '
      NR == 1 && $0 == "---" { in_frontmatter = 1; next }
      in_frontmatter && $0 == "---" { in_frontmatter = 0; next }
      in_frontmatter { next }
      { print }
    ' "$source_file"
  } > "$output_file"
}

sync_codex() {
  local auth_target=""

  mkdir -p "$CODEX_DIR"
  if [[ -L "$CODEX_DIR/auth.json" ]]; then
    auth_target="$(readlink "$CODEX_DIR/auth.json")"
  fi

  log "Replacing Codex config, agents, and managed assets"
  rm -rf \
    "$CODEX_DIR/agents" \
    "$CODEX_DIR/log" \
    "$CODEX_DIR/prompts" \
    "$CODEX_DIR/sessions" \
    "$CODEX_DIR/shell_snapshots" \
    "$CODEX_DIR/skills" \
    "$CODEX_DIR/rules" \
    "$CODEX_DIR/tmp"

  rm -f \
    "$CODEX_DIR/.personality_migration" \
    "$CODEX_DIR/history.jsonl" \
    "$CODEX_DIR/logs_1.sqlite" \
    "$CODEX_DIR/logs_1.sqlite-shm" \
    "$CODEX_DIR/logs_1.sqlite-wal" \
    "$CODEX_DIR/models_cache.json" \
    "$CODEX_DIR/version.json"

  rm -f "$CODEX_DIR"/state_*.sqlite

  copy_file "$ECC_SRC/.codex/config.toml" "$CODEX_DIR/config.toml"
  copy_file "$ECC_SRC/.codex/AGENTS.md" "$CODEX_DIR/AGENTS.md"
  copy_tree "$ECC_SRC/.codex/agents" "$CODEX_DIR/agents"

  mkdir -p "$CODEX_DIR/prompts"
  while IFS= read -r command_file; do
    command_name="$(basename "$command_file" .md)"
    generate_prompt_file "$command_file" "$CODEX_DIR/prompts/ecc-$command_name.md" "$command_name"
  done < <(find "$ECC_SRC/commands" -maxdepth 1 -type f -name '*.md' | sort)

  copy_tree "$ECC_SRC/.agents/skills" "$CODEX_DIR/skills"

  if [[ -n "$auth_target" ]]; then
    ln -sfn "$auth_target" "$CODEX_DIR/auth.json"
  fi
}

sync_agents_skills() {
  log "Replacing .agents skills with ECC skills"
  rm -rf "$AGENTS_DIR"
  mkdir -p "$AGENTS_DIR"
  copy_tree "$ECC_SRC/.agents/skills" "$AGENTS_DIR/skills"
}

sync_claude() {
  log "Updating Claude workspace files with ECC assets"
  mkdir -p "$CLAUDE_DIR"

  rm -f "$CLAUDE_DIR/mcp.json"

  rm -rf \
    "$CLAUDE_DIR/commands" \
    "$CLAUDE_DIR/enterprise" \
    "$CLAUDE_DIR/homunculus" \
    "$CLAUDE_DIR/research" \
    "$CLAUDE_DIR/rules" \
    "$CLAUDE_DIR/skills" \
    "$CLAUDE_DIR/team"

  copy_tree "$ECC_SRC/.claude/commands" "$CLAUDE_DIR/commands"
  copy_tree "$ECC_SRC/.claude/enterprise" "$CLAUDE_DIR/enterprise"
  copy_tree "$ECC_SRC/.claude/homunculus" "$CLAUDE_DIR/homunculus"
  copy_tree "$ECC_SRC/.claude/research" "$CLAUDE_DIR/research"
  copy_tree "$ECC_SRC/.claude/rules" "$CLAUDE_DIR/rules"
  copy_tree "$ECC_SRC/.claude/skills" "$CLAUDE_DIR/skills"
  copy_tree "$ECC_SRC/.claude/team" "$CLAUDE_DIR/team"
  copy_file "$ECC_SRC/.claude/ecc-tools.json" "$CLAUDE_DIR/ecc-tools.json"
  copy_file "$ECC_SRC/.claude/identity.json" "$CLAUDE_DIR/identity.json"
  copy_file "$ECC_SRC/.claude/package-manager.json" "$CLAUDE_DIR/package-manager.json"
}

main() {
  require_dir "$ECC_SRC"
  require_dir "$ECC_SRC/.agents/skills"
  require_dir "$ECC_SRC/.claude"
  require_dir "$ECC_SRC/.codex/agents"
  require_dir "$ECC_SRC/commands"
  require_file "$ECC_SRC/.codex/config.toml"
  require_file "$ECC_SRC/.codex/AGENTS.md"

  sync_agents_skills
  sync_codex
  sync_claude

  log "ECC local setup complete"
}

main "$@"
