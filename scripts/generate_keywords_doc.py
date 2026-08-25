"""Regenerate the keyword tables from allie.lexer.keywords.KEYWORD_ALIASES.

Writes two places, both from the same dict so they cannot disagree:

- `docs/keywords.md` -- the whole file.
- `docs/language-reference.md` -- the block between the
  BEGIN/END GENERATED KEYWORDS markers.

Run after editing KEYWORD_ALIASES:

    python scripts/generate_keywords_doc.py

`tests/unit/test_lexer.py` fails if either output drifts from the code.
"""

from __future__ import annotations

import re
from pathlib import Path

from allie.lexer.keywords import render_keywords_markdown, render_keywords_table

_DOCS = Path(__file__).resolve().parent.parent / "docs"
_BEGIN = (
    "<!-- BEGIN GENERATED KEYWORDS -- do not hand-edit; "
    "run scripts/generate_keywords_doc.py -->"
)
_END = "<!-- END GENERATED KEYWORDS -->"


def render_reference_block() -> str:
    """The generated section as it should appear in
    docs/language-reference.md, markers included."""
    return f"{_BEGIN}\n\n{render_keywords_table()}\n{_END}"


def splice_reference_block(text: str) -> str:
    """Replaces whatever currently sits between the markers. Raises if
    the markers are missing rather than silently appending -- a doc that
    lost its markers needs a human, not a guess."""
    pattern = re.compile(
        re.escape(_BEGIN) + r".*?" + re.escape(_END),
        re.DOTALL,
    )
    if not pattern.search(text):
        raise ValueError(
            "docs/language-reference.md is missing the GENERATED KEYWORDS markers"
        )
    return pattern.sub(lambda _: render_reference_block(), text)


def main() -> None:
    keywords_path = _DOCS / "keywords.md"
    keywords_path.write_text(render_keywords_markdown(), encoding="utf-8")
    print(f"wrote {keywords_path}")

    reference_path = _DOCS / "language-reference.md"
    reference_path.write_text(
        splice_reference_block(reference_path.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    print(f"updated the generated block in {reference_path}")


if __name__ == "__main__":
    main()
