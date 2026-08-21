"""Regenerate docs/keywords.md from allie.lexer.keywords.KEYWORD_ALIASES.

Run after editing KEYWORD_ALIASES:

    python scripts/generate_keywords_doc.py

tests/unit/test_lexer.py fails if the checked-in doc drifts from this.
"""

from __future__ import annotations

from pathlib import Path

from allie.lexer.keywords import render_keywords_markdown


def main() -> None:
    output_path = Path(__file__).resolve().parent.parent / "docs" / "keywords.md"
    output_path.write_text(render_keywords_markdown(), encoding="utf-8")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
