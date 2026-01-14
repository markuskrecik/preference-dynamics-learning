"""
1d CNN model with time series and feature inputs.
"""

import torch
from torch import nn

from preference_dynamics.models.base import PredictorModel
from preference_dynamics.models.schemas import CNN1DFeatConfig


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
        super().__init__(config)

        in_channels = config.in_channels

        self.conv_blocks = nn.ModuleList()
        for out_channels, kernel_size in zip(config.filters, config.kernel_sizes, strict=True):
            conv_block = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding="same",
                ),
                # nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                # nn.Dropout1d(config.dropout),
            )
            self.conv_blocks.append(conv_block)
            in_channels = out_channels

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout(config.dropout),
        )

        self.fc_layers = nn.ModuleList()
        for feature_dim in config.features:
            fc_layer = nn.Sequential(
                nn.LazyLinear(feature_dim),
                nn.LeakyReLU(inplace=True),
                # nn.Dropout(config.dropout),
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
