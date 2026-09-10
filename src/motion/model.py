from typing import Dict

import torch
from torch import Tensor, nn
from torchvision import models


class EfficientNetB0Encoder(nn.Module):

    feature_dim = 1280

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        if hasattr(models, "EfficientNet_B0_Weights"):
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_b0(weights=weights)
        else:
            backbone = models.efficientnet_b0(pretrained=pretrained)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.register_buffer(
            "rgb_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "rgb_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, rgb: Tensor) -> Tensor:
        if not isinstance(rgb, Tensor) or rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("rgb must have shape [N, 3, H, W].")
        if not rgb.is_floating_point():
            raise TypeError("rgb must be floating point, scaled to [0, 1].")
        if rgb.shape[0] == 0 or rgb.shape[2] == 0 or rgb.shape[3] == 0:
            raise ValueError("rgb batch and image dimensions must be positive.")
        normalized = (rgb - self.rgb_mean) / self.rgb_std
        return self.avgpool(self.features(normalized)).flatten(1)


class LocalMotionPrior(nn.Module):

    history_steps = 5
    future_steps = 5
    pose_dim = 3

    def __init__(
        self,
        pretrained: bool = True,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model <= 0 or nhead <= 0 or d_model % nhead:
            raise ValueError("d_model must be positive and divisible by nhead.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        self.image_encoder = EfficientNetB0Encoder(pretrained=pretrained)
        self.coordinate_encoder = nn.Sequential(
            nn.Linear(self.pose_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.history_projection = nn.Sequential(
            nn.Linear(self.image_encoder.feature_dim + 64, d_model),
            nn.LayerNorm(d_model),
        )
        self.future_queries = nn.Parameter(torch.empty(1, self.future_steps, d_model))
        self.temporal_positions = nn.Parameter(
            torch.empty(1, self.history_steps + self.future_steps, d_model)
        )
        nn.init.normal_(self.future_queries, std=0.02)
        nn.init.normal_(self.temporal_positions, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers, norm=nn.LayerNorm(d_model)
        )
        for encoder_layer in self.transformer.layers:
            for parameter in encoder_layer.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)

        self.action_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, self.pose_dim),
        )
        self.feature_head = nn.Linear(d_model, self.image_encoder.feature_dim)

    def forward(self, past_rgb: Tensor, past_positions: Tensor) -> Dict[str, Tensor]:
        self._validate_inputs(past_rgb, past_positions)
        batch, steps, channels, height, width = past_rgb.shape

        visual = self.image_encoder(
            past_rgb.reshape(batch * steps, channels, height, width)
        ).reshape(batch, steps, self.image_encoder.feature_dim)
        coordinates = self.coordinate_encoder(past_positions)
        history = self.history_projection(torch.cat((visual, coordinates), dim=-1))

        queries = self.future_queries.expand(batch, -1, -1)
        tokens = torch.cat((history, queries), dim=1) + self.temporal_positions
        sequence_length = self.history_steps + self.future_steps
        causal_mask = torch.full(
            (sequence_length, sequence_length),
            float("-inf"),
            device=tokens.device,
            dtype=tokens.dtype,
        ).triu(diagonal=1)
        future_states = self.transformer(tokens, mask=causal_mask)[:, steps:]
        return {
            "motion": self.action_head(future_states),
            "features": self.feature_head(future_states),
        }

    def _validate_inputs(self, past_rgb: Tensor, past_positions: Tensor) -> None:
        if not isinstance(past_rgb, Tensor) or past_rgb.ndim != 5:
            raise ValueError("past_rgb must have shape [B, 5, 3, H, W].")
        if past_rgb.shape[1] != self.history_steps or past_rgb.shape[2] != 3:
            raise ValueError("past_rgb requires exactly five frames and three RGB channels.")
        if not isinstance(past_positions, Tensor) or past_positions.ndim != 3:
            raise ValueError("past_positions must have shape [B, 5, 3].")
        expected = (past_rgb.shape[0], self.history_steps, self.pose_dim)
        if tuple(past_positions.shape) != expected:
            raise ValueError("past_positions must match the RGB batch and have shape [B, 5, 3].")
        if not past_rgb.is_floating_point() or not past_positions.is_floating_point():
            raise TypeError("Both inputs must be floating-point tensors.")
        if past_rgb.dtype != past_positions.dtype:
            raise TypeError("RGB and position history must have the same dtype.")
        if past_rgb.device != past_positions.device:
            raise ValueError("RGB and position history must be on the same device.")
        if past_rgb.device != self.future_queries.device:
            raise ValueError("Input tensors and model parameters must be on the same device.")
        if past_rgb.shape[0] == 0 or past_rgb.shape[3] == 0 or past_rgb.shape[4] == 0:
            raise ValueError("RGB batch and image dimensions must be positive.")
