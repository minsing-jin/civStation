from __future__ import annotations

import argparse
from pathlib import Path

from civStation.mcp.client_templates import default_output_path, list_supported_clients, render_client_template


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render or install client templates for the layered MCP server")
    parser.add_argument("--client", required=True, choices=list_supported_clients())
    parser.add_argument("--output", help="Explicit output path. Defaults to the host-specific standard path.")
    parser.add_argument("--write", action="store_true", help="Write the rendered template to disk.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output file when --write is set.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = create_argument_parser()
    args = parser.parse_args(argv)

    content = render_client_template(args.client)
    if not args.write:
        print(content)
        return

    output_path = Path(args.output or default_output_path(args.client))
    if output_path.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite existing file: {output_path}. Re-run with --force if needed.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
