import argparse
import copy
import math
from typing import Dict, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

if __package__:
    from .model import LocalMotionPrior
else:
    from model import LocalMotionPrior


def _check_vo_poses(poses: Tensor, count: Optional[int] = None) -> None:
    if not isinstance(poses, Tensor) or poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("VO poses must be a tensor of shape [N, 4, 4].")
    if count is not None and poses.shape[0] != count:
        raise ValueError("Expected exactly {} time-aligned VO poses.".format(count))
    if not poses.is_floating_point() or not torch.isfinite(poses).all():
        raise ValueError("VO poses must be finite floating-point transforms.")
    bottom = poses.new_tensor([0.0, 0.0, 0.0, 1.0]).expand(poses.shape[0], -1)
    if not torch.allclose(poses[:, 3], bottom, atol=1e-4, rtol=0):
        raise ValueError("VO poses must be homogeneous camera-to-world transforms.")
    rotations = poses[:, :3, :3]
    identity = torch.eye(3, device=poses.device, dtype=poses.dtype).expand_as(rotations)
    if not torch.allclose(rotations.transpose(1, 2) @ rotations, identity, atol=1e-3, rtol=0):
        raise ValueError("VO rotations must be orthonormal.")
    if not torch.allclose(torch.linalg.det(rotations), poses.new_ones(poses.shape[0]), atol=1e-3, rtol=0):
        raise ValueError("VO rotations must have determinant +1.")


def mean_effective_step_length(trajectory_poses: Tensor, epsilon: float = 1e-6) -> Tensor:
    _check_vo_poses(trajectory_poses)
    if trajectory_poses.shape[0] < 2 or epsilon <= 0:
        raise ValueError("At least two poses and a positive epsilon are required.")
    steps = trajectory_poses[1:, :3, 3] - trajectory_poses[:-1, :3, 3]
    lengths = torch.linalg.vector_norm(steps, dim=-1)
    effective = lengths[lengths > epsilon]
    if effective.numel() == 0:
        raise ValueError("No effective translation steps; provide a known positive scale.")
    return effective.mean().detach()


def _rgb_float(rgb: Tensor, count: int) -> Tensor:
    if not isinstance(rgb, Tensor) or rgb.ndim != 4 or rgb.shape[:2] != (count, 3):
        raise ValueError("RGB must have shape [{}, 3, H, W].".format(count))
    if min(rgb.shape[-2:]) < 1:
        raise ValueError("RGB images must have positive spatial dimensions.")
    if rgb.dtype == torch.uint8:
        return rgb.float().div(255)
    if not rgb.is_floating_point() or not torch.isfinite(rgb).all():
        raise ValueError("RGB must be uint8 or finite floating point in [0, 1].")
    if (rgb < 0).any() or (rgb > 1).any():
        raise ValueError("Floating-point RGB must be in [0, 1], before ImageNet normalization.")
    return rgb.float()


def prepare_history(
    past_rgb: Tensor,
    past_vo_poses: Tensor,
    step_scale: Optional[Tensor] = None,
) -> Dict[str, Tensor]:
    rgb = _rgb_float(past_rgb, 5)
    _check_vo_poses(past_vo_poses, count=5)
    poses = past_vo_poses.detach().to(device=rgb.device, dtype=torch.float32)
    if step_scale is None:
        step_scale = mean_effective_step_length(poses)
    scale = torch.as_tensor(step_scale, device=rgb.device, dtype=poses.dtype).detach()
    if scale.numel() != 1 or not torch.isfinite(scale).all() or scale.item() <= 0:
        raise ValueError("step_scale must be a finite, positive scalar.")
    scale = scale.reshape(())
    current_rotation = poses[-1, :3, :3]
    offsets_world = poses[:, :3, 3] - poses[-1, :3, 3]
    local_positions = (offsets_world @ current_rotation) / scale
    return {"past_rgb": rgb, "past_positions": local_positions, "step_scale": scale}


def prepare_training_window(
    rgb_frames: Tensor,
    vo_poses: Tensor,
    trajectory_step_scale: Tensor,
) -> Dict[str, Tensor]:
    rgb = _rgb_float(rgb_frames, 10)
    _check_vo_poses(vo_poses, count=10)
    if trajectory_step_scale is None:
        raise ValueError("Supply the normalization scale of the source training trajectory.")
    history = prepare_history(rgb[:5], vo_poses[:5], trajectory_step_scale)
    poses = vo_poses.detach().to(device=rgb.device, dtype=torch.float32)
    future_steps_world = poses[5:, :3, 3] - poses[4:-1, :3, 3]
    history["future_motion"] = (future_steps_world @ poses[4, :3, :3]) / history["step_scale"]
    history["future_rgb"] = rgb[5:]
    return history


class MotionPriorLoss(nn.Module):

    def __init__(self, model: LocalMotionPrior, action_weight: float = 1.0, feature_weight: float = 1.0):
        super().__init__()
        if not (
            math.isfinite(action_weight) and math.isfinite(feature_weight)
            and action_weight >= 0 and feature_weight >= 0
            and action_weight + feature_weight > 0
        ):
            raise ValueError("Loss weights must be nonnegative, with at least one positive weight.")
        self.action_weight = float(action_weight)
        self.feature_weight = float(feature_weight)
        self.target_encoder = copy.deepcopy(model.image_encoder)
        self.target_encoder.requires_grad_(False)
        self.target_encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_encoder.eval()
        return self

    def forward(self, prediction: Dict[str, Tensor], future_rgb: Tensor, future_motion: Tensor) -> Dict[str, Tensor]:
        motion, features = prediction["motion"], prediction["features"]
        if motion.ndim != 3 or motion.shape[1:] != (5, 3) or future_motion.shape != motion.shape:
            raise ValueError("Predicted and target motion must have shape [B, 5, 3].")
        if future_rgb.ndim != 5 or future_rgb.shape[:3] != (motion.shape[0], 5, 3):
            raise ValueError("Future RGB supervision must have shape [B, 5, 3, H, W].")
        if not torch.isfinite(future_motion).all():
            raise ValueError("Future motion supervision must be finite.")
        self.target_encoder.eval()
        with torch.no_grad():
            target_features = self.target_encoder(future_rgb.detach().flatten(0, 1))
            target_features = target_features.reshape(motion.shape[0], 5, -1)
        if features.shape != target_features.shape:
            raise ValueError("Predicted and target future visual feature shapes must match.")
        action_loss = F.smooth_l1_loss(motion, future_motion.detach())
        feature_loss = F.mse_loss(F.normalize(features, dim=-1), F.normalize(target_features, dim=-1))
        total = self.action_weight * action_loss + self.feature_weight * feature_loss
        return {"total": total, "action": action_loss, "feature": feature_loss}


def train_step(
    model: LocalMotionPrior,
    objective: MotionPriorLoss,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, Tensor],
    max_grad_norm: float = 1.0,
) -> Dict[str, float]:
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive.")
    device = next(model.parameters()).device
    model.train()
    objective.to(device).train()
    optimizer.zero_grad(set_to_none=True)
    prediction = model(batch["past_rgb"].to(device), batch["past_positions"].to(device))
    losses = objective(prediction, batch["future_rgb"].to(device), batch["future_motion"].to(device))
    losses["total"].backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    return {key: value.detach().item() for key, value in losses.items()}


def _smoke_test(device: str) -> None:
    torch.manual_seed(0)
    torch.set_num_threads(2)
    rgb = torch.randint(0, 256, (10, 3, 64, 64), dtype=torch.uint8)
    poses = torch.eye(4).repeat(10, 1, 1)
    poses[:, 0, 3] = torch.arange(10) * 0.2
    sample = prepare_training_window(rgb, poses, mean_effective_step_length(poses))
    batch = {key: value.unsqueeze(0) for key, value in sample.items()}
    model = LocalMotionPrior(pretrained=False).to(device)
    objective = MotionPriorLoss(model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    metrics = train_step(model, objective, optimizer, batch)
    model.eval()
    with torch.no_grad():
        prediction = model(batch["past_rgb"].to(device), batch["past_positions"].to(device))
    print("Synthetic one-step check:", metrics)
    print("Future motion shape:", tuple(prediction["motion"].shape))
    print("Future feature shape:", tuple(prediction["features"].shape))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Run one synthetic update; no weights are downloaded.")
    parser.add_argument("--device", default="cpu", help="Device for the optional smoke test, e.g. cpu or cuda.")
    arguments = parser.parse_args()
    if arguments.smoke_test:
        _smoke_test(arguments.device)
    else:
        parser.print_help()
