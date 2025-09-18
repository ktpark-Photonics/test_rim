"""Configuration loading utilities supporting YAML files and CLI overrides."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, MutableMapping

import yaml


class ConfigError(RuntimeError):
    """Raised when configuration parsing fails."""


@dataclass
class ConfigBundle:
    """Container returned by :func:`load_config`.

    The bundle stores the raw dictionary and exposes helper accessors that make
    it convenient to retrieve nested configuration nodes.
    """

    data: Dict[str, Any]

    def get(self, key: str, default: Any = None) -> Any:
        node = self.data
        for part in key.split("."):
            if isinstance(node, MutableMapping) and part in node:
                node = node[part]
            else:
                return default
        return node

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.data)


def load_yaml(path: Path) -> Dict[str, Any]:
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            content = yaml.safe_load(stream) or {}
    except FileNotFoundError as exc:  # pragma: no cover - configuration missing
        raise ConfigError(f"Configuration file not found: {path}") from exc
    except yaml.YAMLError as exc:  # pragma: no cover - syntax errors
        raise ConfigError(f"Invalid YAML syntax in {path}: {exc}") from exc
    if not isinstance(content, MutableMapping):
        raise ConfigError(f"Expected mapping at top level of {path}")
    return dict(content)


def parse_override(value: str) -> Any:
    """Best effort conversion of CLI override values.

    The parser accepts scalars and falls back to treating the override as a
    string if conversion fails. YAML parsing is used to cover integers, floats,
    booleans, lists and dictionaries.
    """

    try:
        parsed = yaml.safe_load(value)
    except yaml.YAMLError:
        return value
    else:
        if parsed is None and value.lower() not in {"null", "none"}:
            return value
        return parsed


def apply_overrides(config: MutableMapping[str, Any], overrides: Iterable[str]) -> None:
    """Apply dot-separated CLI overrides to a configuration dictionary."""

    for override in overrides:
        if "=" not in override:
            raise ConfigError(f"Invalid override '{override}'. Expected key=value format.")
        key, raw_value = override.split("=", 1)
        value = parse_override(raw_value)
        node = config
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], MutableMapping):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value


def load_config(path: Path, overrides: Iterable[str] | None = None) -> ConfigBundle:
    data = load_yaml(path)
    if overrides:
        apply_overrides(data, overrides)
    return ConfigBundle(data=data)


__all__ = ["ConfigBundle", "ConfigError", "load_config", "apply_overrides"]
