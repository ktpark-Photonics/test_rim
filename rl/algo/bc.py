"""Behavior cloning implementation for imitation pre-training."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from rl.models.policy_head import PolicyHead
from rl.models.vision_backbone import VisionBackbone, VisionBackboneConfig


class DemoDataset(Dataset):
    """Loads recorded demonstrations stored as npz files."""

    def __init__(self, paths: Iterable[Path]) -> None:
        self.samples: List[Tuple[np.ndarray, np.ndarray, int, np.ndarray]] = []
        for path in paths:
            data = np.load(path)
            frames = data["frames"]
            scalars = data["scalars"]
            actions = data["actions"]
            params = data["params"]
            for idx in range(len(actions)):
                image = frames[idx]
                if image.ndim == 4:  # stack, H, W, C
                    image = np.transpose(image, (0, 3, 1, 2))
                    image = image.reshape(-1, image.shape[2], image.shape[3])
                elif image.ndim == 3 and image.shape[-1] in {1, 3}:
                    image = np.transpose(image, (2, 0, 1))
                self.samples.append((image.astype(np.uint8), scalars[idx].astype(np.float32), int(actions[idx]), params[idx].astype(np.float32)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image, scalars, action, params = self.samples[index]
        image = torch.tensor(image, dtype=torch.float32) / 255.0
        scalars = torch.tensor(scalars, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)
        params = torch.tensor(params, dtype=torch.float32)
        return image, scalars, action, params


@dataclass
class BCConfig:
    batch_size: int = 64
    epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 1e-5
    num_workers: int = 0


class BCPolicy(nn.Module):
    def __init__(self, image_shape: Tuple[int, int, int], scalar_dim: int, num_actions: int, param_dim: int) -> None:
        super().__init__()
        channels, height, width = image_shape
        backbone_cfg = VisionBackboneConfig(input_channels=channels, feature_dim=512)
        self.backbone = VisionBackbone(backbone_cfg, input_shape=(channels, height, width))
        self.scalar_encoder = nn.Sequential(nn.Linear(scalar_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(self.backbone.output_dim + 128, 512), nn.ReLU())
        self.policy = PolicyHead(feature_dim=512, num_actions=num_actions, param_dim=param_dim)

    def forward(self, image: torch.Tensor, scalars: torch.Tensor):
        vision = self.backbone(image)
        scalar_feat = self.scalar_encoder(scalars)
        features = self.fusion(torch.cat([vision, scalar_feat], dim=1))
        return self.policy(features)


class BehaviorCloningLearner:
    def __init__(self, config: BCConfig, image_shape: Tuple[int, int, int], scalar_dim: int, num_actions: int, param_dim: int, device: str = "auto") -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = BCPolicy(image_shape, scalar_dim, num_actions, param_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.config = config
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None = None) -> dict:
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        for epoch in range(self.config.epochs):
            train_loss, train_acc = self._run_epoch(train_loader, training=True)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            if val_loader is not None:
                val_loss, val_acc = self._run_epoch(val_loader, training=False)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
        return history

    def _run_epoch(self, loader: DataLoader, training: bool) -> tuple[float, float]:
        if training:
            self.model.train()
        else:
            self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        for image, scalars, action, params in loader:
            image = image.to(self.device)
            scalars = scalars.to(self.device)
            action = action.to(self.device)
            params = params.to(self.device)
            with torch.set_grad_enabled(training):
                output = self.model(image, scalars)
                loss = self.ce_loss(output.logits, action)
                loss = loss + 0.1 * self.mse_loss(output.param_mean, params)
                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
            total_loss += loss.item() * image.size(0)
            predictions = output.logits.argmax(dim=1)
            total_correct += (predictions == action).sum().item()
            total_examples += image.size(0)
        avg_loss = total_loss / max(total_examples, 1)
        accuracy = total_correct / max(total_examples, 1)
        return avg_loss, accuracy

    def act(self, image: np.ndarray, scalars: np.ndarray) -> dict:
        self.model.eval()
        image_tensor = torch.tensor(image, dtype=torch.float32, device=self.device).unsqueeze(0) / 255.0
        scalars_tensor = torch.tensor(scalars, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            output = self.model(image_tensor, scalars_tensor)
            action = torch.distributions.Categorical(logits=output.logits).sample()
            params = torch.tanh(output.param_mean)
        return {"id": int(action.item()), "params": params.squeeze(0).cpu().numpy()}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: Path, map_location: str | torch.device | None = None) -> None:
        state = torch.load(path, map_location=map_location or self.device)
        self.model.load_state_dict(state)


__all__ = ["BehaviorCloningLearner", "BCConfig", "DemoDataset"]
