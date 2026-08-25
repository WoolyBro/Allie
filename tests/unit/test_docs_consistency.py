"""Cross-cutting checklist: docs that describe the implementation are
checked against the implementation.

Phase 9's checklist asks that every grammar rule land in
`docs/grammar.md` alongside the code implementing it. Prose can't be
enforced, but the type list can be -- and it had already drifted:
`PriceSeries` (Phase 4) resolved fine but was missing from the grammar,
while `Series`/`Matrix`/`Vector` were listed as if their type argument
were optional, which the type checker rejects.
"""

from __future__ import annotations

import re
from pathlib import Path

from allie.typechecker.types import _BASE_TYPES_BY_NAME, _GENERIC_TYPES_BY_NAME

_DOCS = Path(__file__).resolve().parents[2] / "docs"


def _grammar_text() -> str:
    return (_DOCS / "grammar.md").read_text(encoding="utf-8")


def _quoted_names(production: str) -> set[str]:
    """Every `"Name"` terminal in a production, following its `|`
    continuation lines."""
    text = _grammar_text()
    match = re.search(rf"^{production}\s+→(.*?)(?=^\w|\n```)", text, re.MULTILINE | re.DOTALL)
    assert match, f"production {production!r} not found in docs/grammar.md"
    return set(re.findall(r'"([A-Za-z_]+)"', match.group(1)))


def test_grammar_base_types_match_the_type_checker() -> None:
    documented = _quoted_names("baseType")
    implemented = set(_BASE_TYPES_BY_NAME)
    assert documented == implemented, (
        f"docs/grammar.md baseType and allie.typechecker.types disagree.\n"
        f"  documented but not resolvable: {sorted(documented - implemented)}\n"
        f"  resolvable but undocumented:   {sorted(implemented - documented)}"
    )


def test_grammar_generic_types_match_the_type_checker() -> None:
    documented = _quoted_names("genericName")
    implemented = set(_GENERIC_TYPES_BY_NAME)
    assert documented == implemented, (
        f"docs/grammar.md genericName and allie.typechecker.types disagree.\n"
        f"  documented but not resolvable: {sorted(documented - implemented)}\n"
        f"  resolvable but undocumented:   {sorted(implemented - documented)}"
    )


def test_generic_and_base_type_names_do_not_overlap() -> None:
    """A name can't be both -- `Series` needs a type argument and
    `Int` can't take one. Overlap would make the grammar ambiguous
    about which form is legal."""
    assert not (set(_BASE_TYPES_BY_NAME) & set(_GENERIC_TYPES_BY_NAME))


def test_referenced_docs_exist() -> None:
    """Every doc the README points at is really there -- a dead link in
    the entry point is the first thing a reader hits."""
    readme = (_DOCS.parent / "README.md").read_text(encoding="utf-8")
    for target in re.findall(r"\]\((docs/[a-z-]+\.md)\)", readme):
        assert (_DOCS.parent / target).exists(), f"README links to missing {target}"
