"""Training entry points for BC, PPO and DAgger pipelines."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from agent.config import load_config
from agent.logger import logger, setup_logging
from rl.algo.bc import BCConfig, BehaviorCloningLearner, DemoDataset
from rl.algo.dagger import DAggerConfig, DAggerTrainer
from rl.algo.ppo import PPOAgent, PPOConfig
from rl.envs.rimworld_wrapper import RimWorldEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RimWorld RL training harness")
    parser.add_argument("--config", type=Path, default=Path("configs/rl.yaml"))
    parser.add_argument("--mode", type=str, default="ppo", choices=["ppo", "bc", "dagger"])
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("data/checkpoints"))
    parser.add_argument("overrides", nargs="*", help="Configuration overrides in key=value format")
    return parser.parse_args()


def make_env(cfg: Dict) -> RimWorldEnv:
    env_cfg = cfg
    if cfg.get("env", {}).get("stub", True):
        rng = np.random.default_rng(seed=cfg.get("train", {}).get("seed", 42))

        def frame_provider():
            width = env_cfg.get("obs", {}).get("width", 320)
            height = env_cfg.get("obs", {}).get("height", 180)
            return rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)

        env = RimWorldEnv(env_cfg, frame_provider=frame_provider)
    else:
        env = RimWorldEnv(env_cfg)
    return env


def run_bc(cfg: Dict, output_dir: Path) -> None:
    demos_dir = Path(cfg.get("data", {}).get("demos", "data/demos"))
    files = sorted(demos_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No demonstration files found in {demos_dir}")
    dataset = DemoDataset(files)
    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len]) if val_len > 0 else (dataset, None)
    bc_cfg = BCConfig(**cfg.get("bc", {}))
    obs_cfg = cfg.get("obs", {})
    image_shape = (
        obs_cfg.get("frame_stack", 4) * (1 if obs_cfg.get("grayscale", False) else 3),
        obs_cfg.get("height", 180),
        obs_cfg.get("width", 320),
    )
    scalar_dim = obs_cfg.get("scalar_features", 8)
    action_cfg = cfg.get("action", {})
    env_actions = cfg.get("actions", []) or []
    num_actions = len(env_actions)
    if num_actions == 0:
        env = make_env(cfg)
        num_actions = len(env.action_cfg.names)
        env.close()
    learner = BehaviorCloningLearner(bc_cfg, image_shape, scalar_dim, num_actions, action_cfg.get("param_dim", 4))
    writer = SummaryWriter(log_dir=output_dir / f"bc-{int(time.time())}")
    train_loader = DataLoader(train_set, batch_size=bc_cfg.batch_size, shuffle=True, num_workers=bc_cfg.num_workers)
    val_loader = None
    if val_len > 0:
        val_loader = DataLoader(val_set, batch_size=bc_cfg.batch_size, shuffle=False, num_workers=bc_cfg.num_workers)
    history = learner.fit(train_loader, val_loader)
    for epoch, (train_loss, train_acc) in enumerate(zip(history["train_loss"], history["train_acc"])):
        writer.add_scalar("bc/train_loss", train_loss, epoch)
        writer.add_scalar("bc/train_acc", train_acc, epoch)
        if history["val_loss"]:
            writer.add_scalar("bc/val_loss", history["val_loss"][epoch], epoch)
            writer.add_scalar("bc/val_acc", history["val_acc"][epoch], epoch)
    checkpoint_path = output_dir / "bc_policy.pt"
    learner.save(checkpoint_path)
    logger.info("Behavior cloning checkpoint saved to %s", checkpoint_path)


def run_ppo(cfg: Dict, output_dir: Path) -> None:
    env = make_env(cfg)
    obs_space = env.observation_space
    action_space = env.action_space
    image_shape = obs_space["image"].shape
    scalar_dim = obs_space["scalars"].shape[0]
    num_actions = action_space["id"].n
    param_dim = action_space["params"].shape[0]
    ppo_cfg = PPOConfig(**cfg.get("ppo", {}))
    agent = PPOAgent(ppo_cfg, image_shape, scalar_dim, num_actions, param_dim)
    writer = SummaryWriter(log_dir=output_dir / f"ppo-{int(time.time())}")
    total_steps = cfg.get("train", {}).get("total_steps", 100_000)
    save_interval = cfg.get("train", {}).get("save_interval", 50_000)
    eval_interval = cfg.get("train", {}).get("eval_interval", 10_000)
    global_step = 0
    episode_return = 0.0
    episode_length = 0
    success_counter = []
    obs, info = env.reset()
    episode_id = 0
    while global_step < total_steps:
        for _ in range(ppo_cfg.steps_per_rollout):
            action_dict = agent.act(obs)
            env_action = {"id": action_dict["id"], "params": action_dict["params"]}
            next_obs, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated
            agent.add_to_buffer(obs, action_dict, reward, done)
            episode_return += reward
            episode_length += 1
            global_step += 1
            writer.add_scalar("train/reward", reward, global_step)
            writer.add_scalar("train/skill_score", info.get("skill_score", 0.0), global_step)
            if done:
                writer.add_scalar("train/episode_return", episode_return, episode_id)
                writer.add_scalar("train/episode_length", episode_length, episode_id)
                success = info.get("success", False)
                success_counter.append(1 if success else 0)
                if len(success_counter) > 100:
                    success_counter.pop(0)
                writer.add_scalar("train/success_rate", np.mean(success_counter) if success_counter else 0.0, episode_id)
                obs, info = env.reset()
                episode_return = 0.0
                episode_length = 0
                episode_id += 1
            else:
                obs = next_obs
            if global_step >= total_steps:
                break
        update_info = agent.update(obs, False)
        for key, value in update_info.items():
            writer.add_scalar(f"ppo/{key}", value, global_step)
        if global_step % save_interval < ppo_cfg.steps_per_rollout:
            path = output_dir / f"ppo-{global_step}.pt"
            torch.save(agent.state_dict(), path)
            logger.info("Saved PPO checkpoint to %s", path)
        if global_step % eval_interval < ppo_cfg.steps_per_rollout:
            writer.add_scalar("train/skill_score_snapshot", info.get("skill_score", 0.0), global_step)
    env.close()


def run_dagger(cfg: Dict, output_dir: Path) -> None:
    env = make_env(cfg)
    obs_space = env.observation_space
    action_space = env.action_space
    image_shape = obs_space["image"].shape
    scalar_dim = obs_space["scalars"].shape[0]
    num_actions = action_space["id"].n
    param_dim = action_space["params"].shape[0]
    bc_cfg = BCConfig(**cfg.get("bc", {}))
    learner = BehaviorCloningLearner(bc_cfg, image_shape, scalar_dim, num_actions, param_dim)
    dagger_cfg = DAggerConfig(**cfg.get("dagger", {}))

    def expert_policy(_: dict) -> dict:
        task = env.current_task
        if task.stage < len(task.steps):
            action_name = task.steps[task.stage].action_name
        else:
            action_name = "NOOP"
        action_id = env.action_cfg.names.index(action_name)
        return {"id": action_id, "params": np.zeros(env.action_cfg.param_dim, dtype=np.float32)}

    trainer = DAggerTrainer(dagger_cfg, learner, expert_policy)
    writer = SummaryWriter(log_dir=output_dir / f"dagger-{int(time.time())}")
    total_steps = cfg.get("train", {}).get("total_steps", 100_000)
    obs, info = env.reset()
    global_step = 0
    while global_step < total_steps:
        policy_action = learner.act(obs["image"], obs["scalars"]) if len(trainer.buffer) > 0 else expert_policy(obs)
        action = trainer.maybe_query_expert(obs, policy_action)
        env_action = {"id": action["id"], "params": action["params"]}
        next_obs, reward, terminated, truncated, info = env.step(env_action)
        global_step += 1
        writer.add_scalar("dagger/reward", reward, global_step)
        if trainer.should_update():
            history = trainer.update_policy()
            if "train_loss" in history:
                writer.add_scalar("dagger/train_loss", history["train_loss"][-1], global_step)
        if terminated or truncated:
            obs, info = env.reset()
        else:
            obs = next_obs
    env.close()
    checkpoint = output_dir / "dagger_policy.pt"
    learner.save(checkpoint)
    logger.info("Saved DAgger-improved policy to %s", checkpoint)


def main() -> None:
    args = parse_args()
    cfg_bundle = load_config(args.config, args.overrides)
    cfg = cfg_bundle.data
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    reward_cfg_path = cfg.get("reward_config", Path("configs/reward.yaml"))
    if "reward" not in cfg and Path(reward_cfg_path).exists():
        cfg["reward"] = load_config(Path(reward_cfg_path)).data
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Training mode: %s", args.mode)
    if args.mode == "bc":
        run_bc(cfg, output_dir)
    elif args.mode == "ppo":
        run_ppo(cfg, output_dir)
    elif args.mode == "dagger":
        run_dagger(cfg, output_dir)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
