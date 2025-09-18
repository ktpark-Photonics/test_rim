"""Minimal behavior tree primitives used to author scripted policies."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable, List


class Status(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"


class Node:
    def tick(self, blackboard: dict) -> Status:
        raise NotImplementedError


@dataclass
class Composite(Node):
    children: List[Node] = field(default_factory=list)

    def add_child(self, node: Node) -> None:
        self.children.append(node)


@dataclass
class Selector(Composite):
    def tick(self, blackboard: dict) -> Status:
        for child in self.children:
            status = child.tick(blackboard)
            if status != Status.FAILURE:
                return status
        return Status.FAILURE


@dataclass
class Sequence(Composite):
    def tick(self, blackboard: dict) -> Status:
        for child in self.children:
            status = child.tick(blackboard)
            if status != Status.SUCCESS:
                return status
        return Status.SUCCESS


@dataclass
class Condition(Node):
    predicate: Callable[[dict], bool]

    def tick(self, blackboard: dict) -> Status:
        return Status.SUCCESS if self.predicate(blackboard) else Status.FAILURE


@dataclass
class Action(Node):
    handler: Callable[[dict], Status]

    def tick(self, blackboard: dict) -> Status:
        return self.handler(blackboard)


class BehaviorTree:
    def __init__(self, root: Node) -> None:
        self.root = root

    def tick(self, blackboard: dict) -> Status:
        return self.root.tick(blackboard)


__all__ = [
    "Status",
    "Node",
    "Selector",
    "Sequence",
    "Condition",
    "Action",
    "BehaviorTree",
]
