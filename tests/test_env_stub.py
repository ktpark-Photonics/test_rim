import numpy as np
import pytest

from rl.envs.rimworld_wrapper import RimWorldEnv


def make_test_env():
    cfg = {
        "env": {"stub": True},
        "obs": {"width": 64, "height": 36, "frame_stack": 2, "scalar_features": 8},
        "reward": {
            "select_colonist": 0.5,
            "open_architect": 0.6,
            "open_zone": 0.7,
            "create_stockpile": 1.2,
            "misclick": 0.1,
            "timeout": 0.5,
            "time_penalty": 0.0,
            "success_bonus": 0.5,
        },
    }
    return RimWorldEnv(cfg)


def test_env_task_progression():
    env = make_test_env()
    obs, info = env.reset()
    assert obs["image"].shape[1:] == (36, 64)
    assert obs["scalars"].shape[0] == env.obs_cfg.scalar_features
    action_sequence = [
        env.action_cfg.names.index("SELECT_COLONIST_0"),
        env.action_cfg.names.index("OPEN_ARCHITECT"),
        env.action_cfg.names.index("OPEN_ZONE"),
        env.action_cfg.names.index("CREATE_STOCKPILE"),
    ]
    total_reward = 0.0
    for action_id in action_sequence:
        obs, reward, terminated, truncated, info = env.step({"id": action_id, "params": np.zeros(env.action_cfg.param_dim)})
        total_reward += reward
        if terminated or truncated:
            env.reset()
    assert total_reward >= 0.5
    assert info["skill_score"] >= 0.0
    env.close()


def test_env_handles_misclick():
    env = make_test_env()
    env.reset()
    obs, reward, *_ = env.step({"id": 0, "params": np.zeros(env.action_cfg.param_dim)})
    assert reward <= 0.0
    env.close()
