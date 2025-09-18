"""Mission/Task definitions and reward shaping utilities."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional


@dataclass
class RewardTable:
    select_colonist: float = 0.2
    open_architect: float = 0.4
    open_zone: float = 0.4
    create_stockpile: float = 1.0
    misclick: float = 0.1
    timeout: float = 0.5
    time_penalty: float = 0.01
    success_bonus: float = 1.0

    def get(self, key: str) -> float:
        return float(getattr(self, key))


@dataclass
class TaskStep:
    action_name: str
    reward_key: str
    description: str


@dataclass
class TaskEvent:
    name: str
    reward: float
    description: str


@dataclass
class TaskResult:
    reward: float
    done: bool
    success: bool
    events: List[TaskEvent]
    stage: int


@dataclass
class Task:
    name: str
    steps: List[TaskStep]
    reward_table: RewardTable
    max_steps: int = 120
    stage: int = 0
    elapsed_steps: int = 0
    success: bool = False

    def reset(self) -> None:
        self.stage = 0
        self.elapsed_steps = 0
        self.success = False

    def step(self, action_name: str, params: Optional[List[float]] = None) -> TaskResult:
        reward = -self.reward_table.time_penalty
        events: List[TaskEvent] = []
        done = False
        success = False
        if self.stage < len(self.steps):
            expected = self.steps[self.stage]
            if action_name == expected.action_name:
                stage_reward = self.reward_table.get(expected.reward_key)
                reward += stage_reward
                events.append(TaskEvent(expected.reward_key, stage_reward, expected.description))
                self.stage += 1
                if self.stage == len(self.steps):
                    success = True
                    done = True
                    bonus = self.reward_table.success_bonus
                    reward += bonus
                    events.append(TaskEvent("success", bonus, f"Task '{self.name}' completed"))
            else:
                penalty = self.reward_table.misclick
                reward -= penalty
                events.append(TaskEvent("misclick", -penalty, f"Expected {expected.action_name}, got {action_name}"))
        else:
            done = True
            success = True

        self.elapsed_steps += 1
        if self.elapsed_steps >= self.max_steps and not done:
            done = True
            penalty = self.reward_table.timeout
            reward -= penalty
            events.append(TaskEvent("timeout", -penalty, f"Task '{self.name}' timed out"))

        if done and success:
            self.success = True
        return TaskResult(reward=reward, done=done, success=success, events=events, stage=self.stage)


@dataclass
class CurriculumRule:
    promotion_threshold: float = 0.8
    demotion_threshold: float = 0.3
    window: int = 20


class CurriculumManager:
    """Tracks task performance and handles automatic promotion/demotion."""

    def __init__(self, tasks: List[Task], rule: CurriculumRule | None = None) -> None:
        self.tasks = tasks
        self.rule = rule or CurriculumRule()
        self.current_index = 0
        self.history: Dict[str, Deque[bool]] = defaultdict(lambda: deque(maxlen=self.rule.window))

    def current_task(self) -> Task:
        return self.tasks[self.current_index]

    def record_outcome(self, task: Task, success: bool) -> None:
        history = self.history[task.name]
        history.append(success)
        rate = sum(history) / len(history)
        if success and rate >= self.rule.promotion_threshold and self.current_index < len(self.tasks) - 1:
            self.current_index += 1
        elif not success and rate <= self.rule.demotion_threshold and self.current_index > 0:
            self.current_index -= 1

    def reset_curriculum(self) -> None:
        self.current_index = 0
        self.history.clear()


def build_default_tasks(action_lookup: Dict[int, str], reward_table: RewardTable) -> List[Task]:
    lookup_by_name = {name: action_id for action_id, name in action_lookup.items()}
    required_actions = ["SELECT_COLONIST_0", "OPEN_ARCHITECT", "OPEN_ZONE", "CREATE_STOCKPILE"]
    for action in required_actions:
        if action not in lookup_by_name:
            raise KeyError(f"Action '{action}' required by default tasks is missing in action specification")

    select = Task(
        name="select_colonist",
        steps=[TaskStep("SELECT_COLONIST_0", "select_colonist", "Select the first colonist")],
        reward_table=reward_table,
        max_steps=30,
    )
    architect = Task(
        name="open_architect",
        steps=[
            TaskStep("SELECT_COLONIST_0", "select_colonist", "Ensure a colonist is selected"),
            TaskStep("OPEN_ARCHITECT", "open_architect", "Open the architect menu"),
        ],
        reward_table=reward_table,
        max_steps=60,
    )
    zone = Task(
        name="open_zone",
        steps=[
            TaskStep("OPEN_ARCHITECT", "open_architect", "Architect menu should be open"),
            TaskStep("OPEN_ZONE", "open_zone", "Open the zone submenu"),
        ],
        reward_table=reward_table,
        max_steps=80,
    )
    stockpile = Task(
        name="create_stockpile",
        steps=[
            TaskStep("OPEN_ZONE", "open_zone", "Ensure zone submenu is active"),
            TaskStep("CREATE_STOCKPILE", "create_stockpile", "Create a stockpile zone"),
        ],
        reward_table=reward_table,
        max_steps=120,
    )
    return [select, architect, zone, stockpile]


__all__ = [
    "RewardTable",
    "Task",
    "TaskResult",
    "TaskEvent",
    "TaskStep",
    "CurriculumRule",
    "CurriculumManager",
    "build_default_tasks",
]
