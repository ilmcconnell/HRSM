#!/usr/bin/env python
"""Generate ``docs/config_reference.md`` from the pydantic config model.

Defaults and types come from ``hrsmpipe.config``; the description of a key is the ``# comment``
that follows its field definition in the source.

Usage: ``pixi run python scripts/gen_config_reference.py`` (``--check`` fails if the page is stale).
"""

from __future__ import annotations

import argparse
import ast
import inspect
import io
import sys
import textwrap
import tokenize
from pathlib import Path

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from hrsmpipe import config

OUT = Path(__file__).resolve().parents[1] / "docs" / "config_reference.md"


def _comments(cls: type[BaseModel]) -> dict[str, str]:
    """``{field: comment}``: the ``#`` comments inside each field's source lines (formatter-proof)."""
    src = textwrap.dedent(inspect.getsource(cls))
    by_line = {
        tok.start[0]: tok.string.lstrip("# ").strip()
        for tok in tokenize.generate_tokens(io.StringIO(src).readline)
        if tok.type == tokenize.COMMENT
    }
    out = {}
    for node in ast.parse(src).body[0].body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            lines = range(node.lineno, node.end_lineno + 1)
            text = " ".join(by_line[i] for i in lines if i in by_line)
            if text:
                out[node.target.id] = text
    return out


def _type_name(annotation) -> str:
    return str(annotation).replace("typing.", "").replace("<class '", "").replace("'>", "")


def _default(field) -> str:
    d = field.default
    if d is PydanticUndefined:
        return "—"
    if isinstance(d, BaseModel):
        return "(table)"
    return f"`{d!r}`" if d != "" else '`""`'


def _rows(prefix: str, cls: type[BaseModel]) -> list[tuple[str, str, str, str]]:
    comments = _comments(cls)
    rows = []
    for name, field in cls.model_fields.items():
        sub = field.annotation
        if isinstance(sub, type) and issubclass(sub, BaseModel):
            rows += _rows(f"{prefix}.{name}", sub)
            continue
        rows.append((f"{prefix}.{name}", _type_name(sub), _default(field), comments.get(name, "")))
    return rows


def render() -> str:
    lines = [
        "# Config reference",
        "",
        "Generated from the pydantic model by `scripts/gen_config_reference.py`; do not edit by hand.",
        "Every key has a default equal to the value used in the original notebook. `hrsmpipe config-schema`",
        "prints the same keys as TOML, and `hrsmpipe init <path>` writes them to a file.",
        "",
        "`site_ID`, `num_hours`, `endtime_s` and `layer_soil_type` are derived at run time and are not config keys.",
        "A `[provenance]` table in an input TOML is ignored and regenerated.",
        "",
    ]
    for name, field in config.RunConfig.model_fields.items():
        cls = field.annotation
        lines += [f"## `[{name}]`", "", "| key | type | default | meaning |", "|---|---|---|---|"]
        for key, typ, default, comment in _rows(name, cls):
            lines.append(f"| `{key}` | `{typ}` | {default} | {comment.replace('|', '\\|')} |")
        lines.append("")
    lines += [
        "## Per-site LGAR overrides",
        "",
        "`[lgar.site_overrides.<site_id>]` may set any of: "
        + ", ".join(f"`{k}`" for k in config.OVERRIDABLE_LGAR_KEYS)
        + ".",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check", action="store_true", help="exit 1 if docs/config_reference.md is out of date"
    )
    args = ap.parse_args()
    text = render()
    if args.check:
        if not OUT.exists() or OUT.read_text() != text:
            print(f"{OUT} is out of date: run scripts/gen_config_reference.py", file=sys.stderr)
            return 1
        return 0
    OUT.write_text(text)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
