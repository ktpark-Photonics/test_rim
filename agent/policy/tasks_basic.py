"""Predefined behavior-tree snippets used for scripted fallback behaviours."""
from __future__ import annotations

from typing import Callable

from .behavior_tree import Action, BehaviorTree, Sequence, Status
from ..action_executor import ActionExecutor


def _make_action(executor: ActionExecutor, macro: str, params_factory: Callable[[dict], list[float]] | None = None) -> Action:
    def handler(blackboard: dict) -> Status:
        params = params_factory(blackboard) if params_factory else []
        executor.execute_macro(macro, params)
        return Status.SUCCESS

    return Action(handler)


def build_architect_open_tree(executor: ActionExecutor) -> BehaviorTree:
    root = Sequence(
        [
            _make_action(executor, "SELECT_COLONIST_0"),
            _make_action(executor, "OPEN_ARCHITECT"),
            _make_action(executor, "OPEN_ZONE"),
        ]
    )
    return BehaviorTree(root)


__all__ = ["build_architect_open_tree"]
