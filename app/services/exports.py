from __future__ import annotations

import html
from pathlib import Path

from ..config import DOWNLOAD_DIR, EXPORT_DIR
from ..utils import ensure_parent, slugify


def markdown_to_html(markdown_text: str) -> str:
    blocks = []
    for chunk in markdown_text.split("\n\n"):
        stripped = chunk.strip()
        if not stripped:
            continue
        if stripped.startswith("# "):
            blocks.append(f"<h1>{html.escape(stripped[2:])}</h1>")
        elif stripped.startswith("## "):
            blocks.append(f"<h2>{html.escape(stripped[3:])}</h2>")
        elif stripped.startswith("- "):
            items = "".join(f"<li>{html.escape(line[2:])}</li>" for line in stripped.splitlines() if line.startswith("- "))
            blocks.append(f"<ul>{items}</ul>")
        else:
            blocks.append(f"<p>{html.escape(stripped)}</p>")
    return "\n".join(blocks)


def export_issue(issue_id: str, markdown_text: str, html_text: str) -> dict[str, str]:
    markdown_path = EXPORT_DIR / f"{issue_id}.md"
    html_path = EXPORT_DIR / f"{issue_id}.html"
    ensure_parent(markdown_path)
    markdown_path.write_text(markdown_text, encoding="utf-8")
    html_path.write_text(html_text, encoding="utf-8")
    return {
        "markdown_path": str(markdown_path),
        "html_path": str(html_path),
    }


def build_download_filename(week_key: str, title: str, fmt: str = "md") -> str:
    extension = "html" if fmt == "html" else "md"
    return f"{week_key}-{slugify(title)}.{extension}"


def save_issue_download_copy(filename: str, content: str) -> Path:
    target_path = DOWNLOAD_DIR / filename
    ensure_parent(target_path)
    target_path.write_text(content, encoding="utf-8")
    return target_path
