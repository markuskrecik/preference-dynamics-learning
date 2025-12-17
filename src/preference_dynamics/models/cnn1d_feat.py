"""
1d CNN model with time series and feature inputs.
"""

from pathlib import Path

import torch
from torch import nn

from preference_dynamics.models.base import PredictorModel
from preference_dynamics.schemas import CNN1DFeatConfig


class CNN1DFeatPredictor(PredictorModel):
    """
    1d CNN model for predicting parameters from time series and features.

    Architecture:
        - Multiple 1d convolutional blocks with ReLU activation
        - Global average pooling to handle variable-length sequences
        - Concatenation of time series and features
        - Fully connected layers to output parameter predictions
    """

    def __init__(self, config: CNN1DFeatConfig) -> None:
        PredictorModel.__init__(self)

        self._config = config

        self.conv_blocks = nn.ModuleList()
        for out_channels, kernel_size in zip(config.filters, config.kernel_sizes, strict=True):
            conv_block = nn.Sequential(
                nn.LazyConv1d(
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding="same",
                ),
                # nn.BatchNorm1d(out_channels),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool1d(2),
                # nn.Dropout1d(config.dropout),
            )
            self.conv_blocks.append(conv_block)

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout(config.dropout),
        )

        self.fc_layers = nn.ModuleList()
        for feature_dim in config.features:
            fc_layer = nn.Sequential(
                nn.LazyLinear(feature_dim),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(config.dropout),
            )
            self.fc_layers.append(fc_layer)

    def forward(self, x: torch.Tensor, x_feat: torch.Tensor) -> torch.Tensor:
        for conv_block in self.conv_blocks:
            x = conv_block(x)  # (batch, filters, time)
        x = self.global_pool(x)  # (batch, filters, 1)
        x = x.squeeze(-1)  # (batch, filters)

        x_feat = torch.nan_to_num(x_feat, nan=0.0)

        x = torch.cat([x, x_feat], dim=-1)  # (batch, filters + feature_dim)

        for fc_layer in self.fc_layers:
            x = fc_layer(x)  # (batch, feature_dim)

        return x

    def save(self, path: str) -> None:
        """
        Save model state and config to disk.

        Saves model state_dict, config, and model_type to a checkpoint file.

        Args:
            path: File path to save model (e.g., 'checkpoints/cnn_best.pt')
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": self.config.model_dump() if hasattr(self.config, "model_dump") else None,
            "model_type": self.config.model_type,
        }

        torch.save(checkpoint, save_path)

    def load(self, path: str) -> None:
        """
        Load model state from disk.

        Loads model state_dict from checkpoint and sets model to eval mode.

        Args:
            path: File path to load model from
        """
        checkpoint = torch.load(path, map_location="cpu")
        self.load_state_dict(checkpoint["model_state_dict"])
        self.eval()

    @property
    def config(self) -> CNN1DFeatConfig:
        """
        Configuration for the CNN1D predictor model.
        """
        return self._config

    @property
    def n_parameters(self) -> int:
        """Return number of trainable parameters in model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
