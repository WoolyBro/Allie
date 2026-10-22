"""Tests for allie/runtime/environment.py."""

from __future__ import annotations

import pytest

from allie.runtime.environment import Environment
from allie.runtime.errors import AllieNameError


def test_shadowing_child_masks_parent() -> None:
    parent = Environment()
    parent.define("x", 1)
    child = Environment(parent=parent)
    child.define("x", 2)

    assert child.get("x") == 2
    assert parent.get("x") == 1


def test_get_walks_parent_chain() -> None:
    parent = Environment()
    parent.define("x", 42)
    child = Environment(parent=parent)

    assert child.get("x") == 42


def test_get_undefined_raises() -> None:
    env = Environment()
    with pytest.raises(AllieNameError):
        env.get("nonexistent")


def test_assign_mutates_outer_frame_from_inner_scope() -> None:
    parent = Environment()
    parent.define("x", 1)
    child = Environment(parent=parent)

    child.assign("x", 99)

    assert parent.get("x") == 99
    assert "x" not in child.vars


def test_assign_undefined_raises_without_creating_global() -> None:
    parent = Environment()
    child = Environment(parent=parent)

    with pytest.raises(AllieNameError):
        child.assign("never_defined", 1)

    assert "never_defined" not in parent.vars
    assert "never_defined" not in child.vars
