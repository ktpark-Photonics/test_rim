"""Hyper-parameter sweep helper using simple grid search."""
from __future__ import annotations

import argparse
import itertools
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hyper-parameter sweeps")
    parser.add_argument("--config", type=Path, default=Path("configs/rl.yaml"))
    parser.add_argument("--mode", type=str, default="ppo", choices=["ppo", "bc", "dagger"])
    parser.add_argument("--output", type=Path, default=Path("data/checkpoints"))
    parser.add_argument("--sweep", action="append", nargs="+", metavar=("KEY", "VALUE"), help="Parameter sweep e.g. ppo.lr 3e-4 1e-4")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_sweeps(entries: Iterable[List[str]]) -> Dict[str, List[str]]:
    sweeps: Dict[str, List[str]] = {}
    if not entries:
        return sweeps
    for entry in entries:
        key, *values = entry
        sweeps[key] = values if values else []
    return sweeps


def run_sweep(config: Path, mode: str, output: Path, sweeps: Dict[str, List[str]], dry_run: bool) -> None:
    if not sweeps:
        sweeps = {"ppo.lr": ["3e-4"], "ppo.clip_ratio": ["0.2"]}
    keys = list(sweeps.keys())
    value_grid = list(itertools.product(*sweeps.values()))
    for values in value_grid:
        overrides = [f"{key}={value}" for key, value in zip(keys, values)]
        run_name = "_".join(v.replace("=", "-") for v in overrides)
        cmd = ["python", "-m", "rl.train", "--config", str(config), "--mode", mode, "--output", str(output / run_name)]
        cmd.extend(overrides)
        if dry_run:
            print("DRY-RUN:", " ".join(cmd))
            continue
        subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    sweeps = parse_sweeps(args.sweep)
    run_sweep(args.config, args.mode, args.output, sweeps, args.dry_run)


if __name__ == "__main__":  # pragma: no cover
    main()
