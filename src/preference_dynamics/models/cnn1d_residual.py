"""
1d CNN model with time series and feature inputs.
"""

import torch
from torch import nn

from preference_dynamics.models.base import PredictorModel
from preference_dynamics.models.schemas import CNN1DResidualConfig


class CNN1DResidualPredictor(PredictorModel):
    """
    1d CNN model for predicting parameters from residual time series and features.

    Architecture:
        - Multiple 1d convolutional blocks with ReLU activation
        - Global average pooling to handle variable-length sequences
        - Concatenation of time series and features
        - Fully connected layers to output parameter predictions
    """

    def __init__(self, config: CNN1DResidualConfig) -> None:
        super().__init__(config)

        in_channels = config.in_channels
        steady_state_channels = in_channels
        out_dim = config.out_dim
        hidden_dims = config.hidden_dims

        self.steady_state_fc = nn.Sequential(
            nn.LazyLinear(in_channels),
            nn.LeakyReLU(inplace=True),
            # nn.Dropout(config.dropout),
        )

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
        for hidden_dim in hidden_dims:
            fc_layer = nn.Sequential(
                nn.LazyLinear(hidden_dim),
                nn.LeakyReLU(inplace=True),
                # nn.Dropout(config.dropout),
            )
            self.fc_layers.append(fc_layer)
        # steady state prediction concatenated at the end
        self.fc_layers.append(nn.LazyLinear(out_dim - steady_state_channels))

    def forward(self, x: torch.Tensor, x_feat: torch.Tensor) -> torch.Tensor:
        x_feat = torch.nan_to_num(x_feat, nan=0.0)
        x_ss = torch.cat([x[..., -1], x_feat], dim=-1)
        c_flat = self.steady_state_fc(x_ss)  # (batch, in_channels)
        c = c_flat.unsqueeze(-1).expand_as(x)  # (batch, in_channels, time)
        r = x - c

        for conv_block in self.conv_blocks:
            r = conv_block(r)  # (batch, filters, time)
        r = self.global_pool(r)  # (batch, filters, 1)
        r = r.squeeze(-1)  # (batch, filters)

        for fc_layer in self.fc_layers:
            r = fc_layer(r)  # (batch, hidden_dim)

        x = torch.cat([r, c_flat], dim=-1)
        return x
