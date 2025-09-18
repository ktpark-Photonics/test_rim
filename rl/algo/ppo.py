"""Proximal Policy Optimization algorithm tailored for RimWorld tasks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from rl.models.policy_head import PolicyHead
from rl.models.value_head import ValueHead
from rl.models.vision_backbone import VisionBackbone, VisionBackboneConfig


@dataclass
class PPOConfig:
    steps_per_rollout: int = 2048
    update_epochs: int = 4
    minibatch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    device: str = "auto"


class ActorCritic(nn.Module):
    def __init__(self, image_shape: Tuple[int, int, int], scalar_dim: int, num_actions: int, param_dim: int) -> None:
        super().__init__()
        channels, height, width = image_shape
        backbone_cfg = VisionBackboneConfig(input_channels=channels, feature_dim=512)
        self.backbone = VisionBackbone(backbone_cfg, (channels, height, width))
        self.scalar_encoder = nn.Sequential(nn.Linear(scalar_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(self.backbone.output_dim + 128, 512), nn.ReLU())
        self.policy = PolicyHead(feature_dim=512, num_actions=num_actions, param_dim=param_dim)
        self.value = ValueHead(feature_dim=512)

    def forward(self, image: torch.Tensor, scalars: torch.Tensor):
        vision = self.backbone(image)
        scalar_feat = self.scalar_encoder(scalars)
        features = self.fusion(torch.cat([vision, scalar_feat], dim=1))
        policy_output = self.policy(features)
        value = self.value(features)
        return policy_output, value


class RolloutBuffer:
    def __init__(self) -> None:
        self.images = []
        self.scalars = []
        self.actions = []
        self.params = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.advantages = None
        self.returns = None

    def add(self, image, scalars, action, params, log_prob, value, reward, done) -> None:
        self.images.append(torch.tensor(image, dtype=torch.float32) / 255.0)
        self.scalars.append(torch.tensor(scalars, dtype=torch.float32))
        self.actions.append(torch.tensor(action, dtype=torch.long))
        self.params.append(torch.tensor(params, dtype=torch.float32))
        self.log_probs.append(torch.tensor(log_prob, dtype=torch.float32))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(float(value))

    def compute_returns_and_advantages(self, last_value: float, gamma: float, gae_lambda: float) -> None:
        advantages = []
        gae = 0.0
        values = self.values + [float(last_value)]
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + gamma * values[step + 1] * (1.0 - float(self.dones[step])) - values[step]
            gae = delta + gamma * gae_lambda * (1.0 - float(self.dones[step])) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, self.values)]
        self.advantages = torch.tensor(advantages, dtype=torch.float32)
        self.returns = torch.tensor(returns, dtype=torch.float32)

    def get_batches(self, batch_size: int):
        count = len(self.actions)
        if count == 0:
            return
        images = torch.stack(self.images)
        scalars = torch.stack(self.scalars)
        actions = torch.stack(self.actions)
        params = torch.stack(self.params)
        log_probs = torch.stack(self.log_probs)
        advantages = self.advantages
        returns = self.returns
        count = len(self.actions)
        indices = torch.randperm(count)
        for start in range(0, count, batch_size):
            end = min(start + batch_size, count)
            batch_idx = indices[start:end]
            yield (
                images[batch_idx],
                scalars[batch_idx],
                actions[batch_idx],
                params[batch_idx],
                log_probs[batch_idx],
                advantages[batch_idx],
                returns[batch_idx],
            )

    def clear(self) -> None:
        self.__init__()


class PPOAgent:
    def __init__(self, config: PPOConfig, image_shape: Tuple[int, int, int], scalar_dim: int, num_actions: int, param_dim: int) -> None:
        if config.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(config.device)
        self.device = device
        self.config = config
        self.model = ActorCritic(image_shape, scalar_dim, num_actions, param_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.buffer = RolloutBuffer()
        self.num_actions = num_actions
        self.param_dim = param_dim

    def act(self, observation: Dict[str, np.ndarray], deterministic: bool = False) -> Dict:
        image = torch.tensor(observation["image"], dtype=torch.float32, device=self.device).unsqueeze(0) / 255.0
        scalars = torch.tensor(observation["scalars"], dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            output, value = self.model(image, scalars)
        action_dist = Categorical(logits=output.logits)
        param_std = output.param_log_std.exp()
        param_dist = Normal(output.param_mean, param_std)
        if deterministic:
            action = torch.argmax(action_dist.probs)
            params = output.param_mean
        else:
            action = action_dist.sample()
            params = param_dist.sample()
        log_prob = action_dist.log_prob(action) + param_dist.log_prob(params).sum(-1)
        return {
            "id": int(action.item()),
            "params": torch.tanh(params).squeeze(0).cpu().numpy(),
            "value": value.squeeze(0).cpu().item(),
            "log_prob": log_prob.squeeze(0).cpu().item(),
        }

    def add_to_buffer(self, observation: Dict[str, np.ndarray], action: Dict, reward: float, done: bool) -> None:
        self.buffer.add(
            observation["image"],
            observation["scalars"],
            action["id"],
            action["params"],
            action["log_prob"],
            action["value"],
            reward,
            done,
        )

    def update(self, last_observation: Dict[str, np.ndarray], last_done: bool) -> Dict[str, float]:
        with torch.no_grad():
            last = torch.tensor(last_observation["image"], dtype=torch.float32, device=self.device).unsqueeze(0) / 255.0
            last_scalar = torch.tensor(last_observation["scalars"], dtype=torch.float32, device=self.device).unsqueeze(0)
            _, last_value = self.model(last, last_scalar)
        self.buffer.compute_returns_and_advantages(last_value.item(), self.config.gamma, self.config.gae_lambda)
        advantages = self.buffer.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.buffer.advantages = advantages
        info = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        total_updates = 0
        for epoch in range(self.config.update_epochs):
            for batch in self.buffer.get_batches(self.config.minibatch_size):
                images, scalars, actions, params, old_log_probs, advs, returns = batch
                images = images.to(self.device)
                scalars = scalars.to(self.device)
                actions = actions.to(self.device)
                params = params.to(self.device)
                old_log_probs = old_log_probs.to(self.device)
                advs = advs.to(self.device)
                returns = returns.to(self.device)

                output, values = self.model(images, scalars)
                action_dist = Categorical(logits=output.logits)
                param_std = output.param_log_std.exp()
                param_dist = Normal(output.param_mean, param_std)
                log_probs = action_dist.log_prob(actions) + param_dist.log_prob(params).sum(-1)
                ratios = torch.exp(log_probs - old_log_probs)
                surr1 = ratios * advs
                surr2 = torch.clamp(ratios, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * advs
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values.squeeze(-1), returns)
                entropy = action_dist.entropy().mean() + param_dist.entropy().sum(-1).mean()
                loss = policy_loss + self.config.vf_coef * value_loss - self.config.ent_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                info["policy_loss"] += policy_loss.item()
                info["value_loss"] += value_loss.item()
                info["entropy"] += entropy.item()
                total_updates += 1
        if total_updates > 0:
            info = {k: v / total_updates for k, v in info.items()}
        self.buffer.clear()
        return info

    def state_dict(self):
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state):
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])


__all__ = ["PPOAgent", "PPOConfig", "RolloutBuffer"]
