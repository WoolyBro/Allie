"""The package metadata in pyproject.toml matches the code.

`[tool.setuptools] packages` is an explicit list, so it can drift from
the tree in either direction: a new package left out of the wheel, or a
deleted one still listed (four empty placeholder packages sat there for
a while). And the runtime dependencies are what `allie` imports, no more.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import allie

_ROOT = Path(__file__).resolve().parents[2]
_PROJECT = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _packages_on_disk() -> set[str]:
    return {
        ".".join(init.parent.relative_to(_ROOT).parts)
        for init in (_ROOT / "allie").rglob("__init__.py")
    }


def _imported_top_level_modules() -> set[str]:
    modules: set[str] = set()
    for path in (_ROOT / "allie").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                modules.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.add(node.module.split(".")[0])
    return modules


def test_the_listed_packages_are_exactly_the_packages_in_the_tree() -> None:
    assert set(_PROJECT["tool"]["setuptools"]["packages"]) == _packages_on_disk()


def test_runtime_dependencies_are_what_allie_imports() -> None:
    declared = {
        requirement.split(">")[0].split("=")[0].split("<")[0].strip()
        for requirement in _PROJECT["project"]["dependencies"]
    }
    third_party = {"numpy", "llvmlite", "pandas"}  # every one the project has ever used
    assert declared == _imported_top_level_modules() & third_party


def test_the_version_is_the_same_everywhere() -> None:
    assert allie.__version__ == _PROJECT["project"]["version"]
