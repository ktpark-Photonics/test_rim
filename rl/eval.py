"""Evaluation utilities for trained RimWorld agents."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from agent.config import load_config
from agent.logger import logger, setup_logging
from rl.algo.bc import BCConfig, BehaviorCloningLearner
from rl.algo.ppo import PPOAgent, PPOConfig
from rl.algo.replay import EpisodeRecorder
from rl.train import make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RimWorld agents")
    parser.add_argument("--config", type=Path, default=Path("configs/rl.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=["ppo", "bc", "dagger"], default="ppo")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--record", type=Path, default=None)
    return parser.parse_args()


def evaluate(cfg: Dict, checkpoint: Path, mode: str, episodes: int, record_dir: Path | None) -> Dict[str, float]:
    env = make_env(cfg)
    obs_space = env.observation_space
    action_space = env.action_space
    image_shape = obs_space["image"].shape
    scalar_dim = obs_space["scalars"].shape[0]
    num_actions = action_space["id"].n
    param_dim = action_space["params"].shape[0]
    if mode == "ppo":
        agent = PPOAgent(PPOConfig(**cfg.get("ppo", {})), image_shape, scalar_dim, num_actions, param_dim)
        state = torch.load(checkpoint, map_location=agent.device)
        agent.load_state_dict(state)
        act_fn = lambda obs: agent.act(obs, deterministic=True)
    else:
        learner = BehaviorCloningLearner(BCConfig(**cfg.get("bc", {})), image_shape, scalar_dim, num_actions, param_dim)
        learner.load(checkpoint)
        act_fn = lambda obs: learner.act(obs["image"], obs["scalars"])
    recorder = EpisodeRecorder(record_dir) if record_dir else None
    returns = []
    lengths = []
    successes = []
    skill_scores = []
    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        episode_return = 0.0
        steps = 0
        while not done:
            action = act_fn(obs)
            env_action = {"id": action["id"], "params": action["params"]}
            next_obs, reward, terminated, truncated, info = env.step(env_action)
            episode_return += reward
            steps += 1
            if recorder:
                recorder.record(obs["image"], action["id"], action["params"], reward)
            done = terminated or truncated
            obs = next_obs
        returns.append(episode_return)
        lengths.append(steps)
        successes.append(1 if info.get("success", False) else 0)
        skill_scores.append(info.get("skill_score", 0.0))
        if recorder:
            recorder.save(f"episode-{episode}")
    env.close()
    metrics = {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "length_mean": float(np.mean(lengths)),
        "success_rate": float(np.mean(successes) if successes else 0.0),
        "skill_score": float(np.mean(skill_scores) if skill_scores else 0.0),
    }
    return metrics


def main() -> None:
    args = parse_args()
    cfg_bundle = load_config(args.config)
    cfg = cfg_bundle.data
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    record_dir = args.record
    if record_dir:
        record_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate(cfg, args.checkpoint, args.mode, args.episodes, record_dir)
    logger.info("Evaluation results: %s", metrics)


if __name__ == "__main__":  # pragma: no cover
    main()
