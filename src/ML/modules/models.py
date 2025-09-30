"""
Fourier Neural Operator Network (FNOnet) for processing field and scalar inputs.

This module implements a neural network architecture based on Fourier Neural Operators
for handling both field (spatial) and scalar inputs, with optional separate output heads
for field and scalar predictions.
"""

from typing import Any

import torch
import torch.nn as nn
from neuralop.models import FNO


class FNOnet(nn.Module):
    """A Fourier Neural Operator network for field and scalar data processing.

    This network combines field inputs (spatial data) and scalar inputs through
    a Fourier Neural Operator backbone, with optional separate heads for field
    and scalar outputs. Includes residual connections for improved training.

    Args:
        field_inputs_n: Number of field input channels.
        scalar_inputs_n: Number of scalar input channels.
        field_outputs_n: Number of field output channels. Defaults to 0.
        scalar_outputs_n: Number of scalar output channels. Defaults to 0.
        **kwargs: Additional keyword arguments for network hyperparameters:
            - n_modes_y: Number of Fourier modes in y-direction. Defaults to 16.
            - n_modes_x: Number of Fourier modes in x-direction. Defaults to 16.
            - hidden_channels: Number of hidden channels in FNO. Defaults to 64.
            - n_layers: Number of FNO layers. Defaults to 4.
            - lifting_channels: Channels in lifting operator. Defaults to 32.
            - projection_channels: Channels in projection operator. Defaults to 32.

    Attributes:
        field_inputs_n: Number of field input channels.
        scalar_inputs_n: Number of scalar input channels.
        field_outputs_n: Number of field output channels.
        scalar_outputs_n: Number of scalar output channels.
        n_modes_y: Number of Fourier modes in y-direction.
        n_modes_x: Number of Fourier modes in x-direction.
        hidden_channels: Number of hidden channels in FNO backbone.
        n_layers: Number of layers in FNO backbone.
        lifting_channels: Number of channels in lifting operator.
        projection_channels: Number of channels in projection operator.
        fno: The FNO backbone network.
        field_head: Optional convolutional head for field outputs.
        scalar_head: Optional head for scalar outputs with global pooling.
        residual_conv: 1x1 convolution for matching residual connection dimensions.

    Example:
        >>> # Create network with 3 field inputs, 2 scalar inputs, 1 field output
        >>> model = FNOnet(
        ...     field_inputs_n=3,
        ...     scalar_inputs_n=2,
        ...     field_outputs_n=1,
        ...     scalar_outputs_n=0,
        ...     n_modes_x=32,
        ...     n_modes_y=32
        ... )
        >>>
        >>> # Prepare inputs
        >>> field_inputs = [torch.randn(4, 64, 64) for _ in range(3)]
        >>> scalar_inputs = [torch.randn(4) for _ in range(2)]
        >>>
        >>> # Forward pass
        >>> field_out, scalar_out = model((field_inputs, scalar_inputs))
    """

    def __init__(
        self,
        field_inputs_n: int,
        scalar_inputs_n: int,
        field_outputs_n: int = 0,
        scalar_outputs_n: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize the FNOnet model.

        Args:
            field_inputs_n: Number of field input channels.
            scalar_inputs_n: Number of scalar input channels.
            field_outputs_n: Number of field output channels.
            scalar_outputs_n: Number of scalar output channels.
            **kwargs: Additional hyperparameters for the FNO backbone.
        """
        super().__init__()

        # Store input/output dimensions
        self.field_inputs_n = field_inputs_n
        self.scalar_inputs_n = scalar_inputs_n
        self.field_outputs_n = field_outputs_n
        self.scalar_outputs_n = scalar_outputs_n

        # Unpack additional network hyperparameters from kwargs and set defaults
        self.n_modes_y: int = kwargs.get("n_modes_y", 16)
        self.n_modes_x: int = kwargs.get("n_modes_x", 16)
        self.hidden_channels: int = kwargs.get("hidden_channels", 64)
        self.n_layers: int = kwargs.get("n_layers", 4)
        self.lifting_channels: int = kwargs.get("lifting_channels", 32)
        self.projection_channels: int = kwargs.get("projection_channels", 32)

        # FNO Backbone (Fourier Neural Operator)
        self.fno = FNO(
            n_modes=(self.n_modes_y, self.n_modes_x),
            max_n_modes=(self.n_modes_y, self.n_modes_x),
            hidden_channels=self.hidden_channels,
            in_channels=self.field_inputs_n + self.scalar_inputs_n,
            out_channels=self.field_outputs_n + self.scalar_outputs_n,
            n_layers=self.n_layers,
            lifting_channels=self.lifting_channels,
            projection_channels=self.projection_channels,
        )

        # Optional head for field outputs
        if self.field_outputs_n > 0:
            self.field_head = nn.Conv2d(
                in_channels=self.field_outputs_n + self.scalar_outputs_n,
                out_channels=self.field_outputs_n,
                kernel_size=1,
            )

        # Optional head for scalar outputs
        if self.scalar_outputs_n > 0:
            self.scalar_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(
                    self.field_outputs_n + self.scalar_outputs_n, self.scalar_outputs_n
                ),
            )

        # 1x1 convolution for matching output channels in residual connection
        self.residual_conv = nn.Conv2d(
            in_channels=self.field_inputs_n + self.scalar_inputs_n,
            out_channels=self.field_outputs_n + self.scalar_outputs_n,
            kernel_size=1,
        )

    def forward(
        self, inputs: tuple[list[torch.Tensor], list[torch.Tensor]]
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Forward pass through the FNOnet.

        Processes field and scalar inputs through the FNO backbone and optional
        output heads. Includes residual connections for improved gradient flow.

        Args:
            inputs: Tuple containing:
                - field_inputs: List of field tensors, each of shape (batch_size, height, width).
                - scalar_inputs: List of scalar tensors, each of shape (batch_size,).

        Returns:
            Tuple containing:
                - field_output: Field predictions of shape (batch_size, field_outputs_n, height, width)
                  or None if field_outputs_n == 0.
                - scalar_output: Scalar predictions of shape (batch_size, scalar_outputs_n)
                  or None if scalar_outputs_n == 0.

        Raises:
            RuntimeError: If field input tensors have inconsistent spatial dimensions.
            ValueError: If batch sizes are inconsistent across inputs.
        """
        field_inputs, scalar_inputs = inputs
        batch_size = field_inputs[0].shape[0]

        # Stack field inputs along the channel dimension
        # Shape: (batch_size, field_inputs_n, height, width)
        x_fields = torch.stack(field_inputs, dim=1)

        # Expand scalar inputs to match field dimensions and concatenate
        if scalar_inputs:
            h, w = x_fields.shape[2:]  # Get spatial dimensions
            scalar_expanded = [
                s.view(batch_size, 1, 1, 1).expand(-1, -1, h, w) for s in scalar_inputs
            ]
            x_combined = torch.cat([x_fields] + scalar_expanded, dim=1)
        else:
            x_combined = x_fields

        # Save input for residual connection
        residual = x_combined

        # Forward pass through the FNO backbone
        fno_output = self.fno(x_combined)

        # Match shapes for the residual connection
        if fno_output.shape != residual.shape:
            # Use a 1x1 convolution to match the output channels
            residual = self.residual_conv(residual)

        # Add residual connection
        fno_output = fno_output + residual

        # Apply output heads for field and scalar outputs
        field_output: torch.Tensor | None = (
            self.field_head(fno_output) if self.field_outputs_n > 0 else None
        )
        scalar_output: torch.Tensor | None = (
            self.scalar_head(fno_output) if self.scalar_outputs_n > 0 else None
        )

        return field_output, scalar_output

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters in the model.

        Returns:
            Total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_info(self) -> dict[str, Any]:
        """Get comprehensive model information.

        Returns:
            Dictionary containing model configuration and parameter counts.
        """
        return {
            "field_inputs_n": self.field_inputs_n,
            "scalar_inputs_n": self.scalar_inputs_n,
            "field_outputs_n": self.field_outputs_n,
            "scalar_outputs_n": self.scalar_outputs_n,
            "n_modes_x": self.n_modes_x,
            "n_modes_y": self.n_modes_y,
            "hidden_channels": self.hidden_channels,
            "n_layers": self.n_layers,
            "lifting_channels": self.lifting_channels,
            "projection_channels": self.projection_channels,
            "total_parameters": self.get_num_parameters(),
        }
