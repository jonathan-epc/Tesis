from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.models import FNO, SFNO, TFNO, UNO


class FNOnet(nn.Module):
    def __init__(
        self,
        field_inputs_n: int,
        scalar_inputs_n: int,
        field_outputs_n: int = 0,
        scalar_outputs_n: int = 0,
        **kwargs,
    ):
        super(FNOnet, self).__init__()
        self.field_inputs_n = field_inputs_n
        self.scalar_inputs_n = scalar_inputs_n
        self.field_outputs_n = field_outputs_n
        self.scalar_outputs_n = scalar_outputs_n

        # Unpack additional network hyperparameters from kwargs and set defaults if not present
        self.n_modes_y = kwargs.get("n_modes_y", 16)
        self.n_modes_x = kwargs.get("n_modes_x", 16)
        self.hidden_channels = kwargs.get("hidden_channels", 64)
        self.n_layers = kwargs.get("n_layers", 4)
        self.lifting_channels = kwargs.get("lifting_channels", 32)
        self.projection_channels = kwargs.get("projection_channels", 32)

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

        # 1x1 convolution for matching output channels
        self.residual_conv = nn.Conv2d(
            in_channels=self.field_inputs_n + self.scalar_inputs_n,
            out_channels=self.field_outputs_n + self.scalar_outputs_n,
            kernel_size=1,
        )

    def forward(
        self, inputs: Tuple[List[torch.Tensor], List[torch.Tensor]]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        field_inputs, scalar_inputs = inputs
        batch_size = field_inputs[0].shape[0]

        # Stack field inputs along the channel dimension
        x_fields = torch.stack(field_inputs, dim=1)

        # Expand scalar inputs to match field dimensions and concatenate along the channel dimension
        if scalar_inputs:
            h, w = x_fields.shape[
                2:
            ]  # Adjusted to [batch, channels, height, width] format
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

        # Add residual connection (ensuring shapes match)
        fno_output = fno_output + residual

        # Output heads for field and scalar outputs
        field_output = self.field_head(fno_output) if self.field_outputs_n > 0 else None
        scalar_output = (
            self.scalar_head(fno_output) if self.scalar_outputs_n > 0 else None
        )

        return field_output, scalar_output


# ### Red neuronal

# #### Capa de Fourier


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat
            )
        )
        self.weights2 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat
            )
        )

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


# #### Perceptrón multicapa


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x


# #### Red completa


class FNOold(nn.Module):
    def __init__(
        self,
        inputs_n,
        outputs_n,
        numpoints_x,
        numpoints_y,
        modes1=4,
        modes2=4,
        width=32,
        input_channels=1,
        output_channels=6,
        **kwargs
    ):
        super(FNOold, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desired channel dimension by self.p (Linear layer).
        2. 4 layers of the integral operators u' = (W + K)(u), where:
            - W is defined by self.w (Conv2d layer).
            - K is defined by self.conv (SpectralConv2d layer).
        3. Project from the channel space to the output space by self.q (MLP layer).
        
        Input: the solution of the coefficient function and locations (a(x, y), x, y).
        Input shape: (batchsize, x=s, y=s, c=inputs_n + 1).
        Output: the solution.
        Output shape: (batchsize, x=s, y=s, c=outputs_n).
        """
        self.inputs_n = inputs_n
        self.outputs_n = outputs_n
        self.numpoints_x = numpoints_x
        self.numpoints_y = numpoints_y

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9  # Pad the domain if the input is non-periodic

        # Adjusting input and output channels
        self.input_channels = 5  # Field + parameters (1 field + 4 parameters in this case)
        self.output_channels = output_channels

        # Linear layer to lift the input to the desired channel dimension
        self.p = nn.Linear(self.input_channels + 2, self.width)  # Include 2 channels for grid (x, y)

        # Spectral convolution layers (Fourier layers)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        # MLP layers after spectral convolutions
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)

        # 1x1 Conv layers to handle local interactions (W layers)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        # MLP for projecting the output to the desired output channels
        self.q = MLP(self.width, self.output_channels, self.width * 4)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x_params, x_field = x
        x_field = x_field.unsqueeze(1)  # Add channel dimension to the field (batch, 1, H, W)
        
        # Expand the parameter tensor across the spatial dimensions
        x_params = x_params.view(x_params.size(0), self.inputs_n, 1, 1).expand(-1, -1, self.numpoints_y, self.numpoints_x)
        
        # Concatenate parameters and field along the channel dimension
        x_combined = torch.cat([x_field, x_params], dim=1)  # (batch, inputs_n + 1, H, W)
        
        # Generate spatial grid and concatenate it to the input
        grid = self.get_grid(x_combined.shape, x_combined.device)  # (batch, H, W, 2)
        grid = grid.permute(0, 3, 1, 2)  # (batch, 2, H, W) to match x_combined
        x_combined = torch.cat((x_combined, grid), dim=1)  # (batch, channels + 2, H, W)

        # Lift input to the higher-dimensional space using the linear layer
        x = x_combined.permute(0, 2, 3, 1)  # (batch, H, W, channels)
        x = self.p(x)  # Apply linear layer
        x = x.permute(0, 3, 1, 2)  # Back to (batch, channels, H, W)

        # Padding before convolution
        x = F.pad(x, [0, self.padding, 0, self.padding])

        # Apply four layers of spectral convolutions followed by MLPs
        
        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        # Remove padding
        x = x[..., : -self.padding, : -self.padding]

        # Project the output to the desired number of output channels
        x = self.q(x)
        return x

    def get_grid(self, shape, device):
        batchsize, _, height, width = shape  # height = numpoints_y, width = numpoints_x
        gridx = torch.linspace(0, 1, width, device=device).reshape(1, 1, width, 1).repeat(batchsize, height, 1, 1)
        gridy = torch.linspace(0, 1, height, device=device).reshape(1, height, 1, 1).repeat(batchsize, 1, width, 1)
        return torch.cat((gridx, gridy), dim=-1)
