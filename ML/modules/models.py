import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.models import FNO, TFNO, SFNO, UNO
from typing import Tuple

class FNOnet(nn.Module):
    def __init__(
        self,
        parameters_n,
        variables_n,
        **kwargs
    ):
        super(FNOnet, self).__init__()
        
        self.parameters_n = parameters_n
        self.variables_n = variables_n
        
        for key, value in kwargs.items():
            setattr(self, key, value)

        # FNO for 2D field
        self.fno = FNO(
            n_modes=(self.n_modes_y, self.n_modes_x),
            hidden_channels=self.hidden_channels,
            in_channels=self.parameters_n,  # 1 for the field channel and n for the embedded parameters
            out_channels=self.variables_n,
            n_layers=self.n_layers,
            lifting_channels=self.lifting_channels,
            projection_channels=self.projection_channels
        )

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x_params, x_field = x
        x_field = x_field.unsqueeze(1)
        batch_size, _, height, width = x_field.shape
        x_params = x_params.view(batch_size, self.parameters_n-1, 1, 1).expand(-1, -1, height, width)
        # Concatenate parameters and field along channel dimension
        x_combined = torch.cat([x_field, x_params], dim=1)

        # Forward pass through FNO
        output = self.fno(x_combined)
        output = output + x_field


        return output

class FNOnet2(nn.Module):
    def __init__(
        self,
        parameters_n,
        variables_n,
        **kwargs
    ):
        super(FNOnet2, self).__init__()
        
        self.parameters_n = parameters_n
        self.variables_n = variables_n
        
        for key, value in kwargs.items():
            setattr(self, key, value)

        # FNO for 2D field
        self.fno = TFNO(
            n_modes=(self.n_modes_y, self.n_modes_x),
            hidden_channels=self.hidden_channels,
            in_channels=self.parameters_n,  # 1 for the field channel and n for the embedded parameters
            out_channels=self.variables_n,
            n_layers=self.n_layers,
            lifting_channels=self.lifting_channels,
            projection_channels=self.projection_channels,
            factorization='tucker',
            implementation='factorized',
            rank=0.05
        )
        
        # Additional linear layer to learn linear combinations of the FNO outputs
        self.linear_combination = nn.Conv2d(
            in_channels=self.variables_n,  # Input channels = number of variables
            out_channels=self.variables_n,  # Output channels = number of variables (unchanged)
            kernel_size=1  # A 1x1 convolution for channel-wise transformation (essentially a linear layer)
        )

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x_params, x_field = x
        x_field = x_field.unsqueeze(1)
        batch_size, _, height, width = x_field.shape
        x_params = x_params.view(batch_size, self.parameters_n-1, 1, 1).expand(-1, -1, height, width)
        # Concatenate parameters and field along channel dimension
        x_combined = torch.cat([x_field, x_params], dim=1)

        # Forward pass through FNO
        output = self.fno(x_combined)
        output = output + x_field

        # Pass through the additional linear combination layer
        output = self.linear_combination(output)

        return output


class FNOneti(nn.Module):
    def __init__(
        self,
        parameters_n,
        variables_n,
        **kwargs
    ):
        super(FNOneti, self).__init__()
        
        self.parameters_n = parameters_n
        self.variables_n = variables_n
        
        for key, value in kwargs.items():
            setattr(self, key, value)

        # FNO for 2D field
        self.fno = FNO(
            n_modes=(self.n_modes_y, self.n_modes_x),
            hidden_channels=self.hidden_channels,
            in_channels=self.variables_n,
            out_channels=self.parameters_n,  # 1 for the field channel and 1 for the embedded parameters
            n_layers=self.n_layers,
            lifting_channels=self.lifting_channels,
            projection_channels=self.projection_channels
        )

    def forward(self, x: torch.Tensor) -> list:
        x = self.fno(x)
        
        B = x[:, -1, :, :]  # Take the last channel as the 2D tensor (batch_size, height, width)
        
        parameters = x[:, :-1, :, :]  # Take all channels except the last (batch_size, parameters_n, height, width)
        parameters = torch.mean(parameters, dim=[2, 3])  # Reduce height and width to get a 1D tensor (batch_size, parameters_n)
        
        return [parameters, B]  # Return a list with 1D and 2D tensors


class UNetNet(nn.Module):
    def __init__(self, parameters_n, variables_n, numpoints_x, numpoints_y, **kwargs):
        super(UNetNet, self).__init__()

        self.parameters_n = parameters_n
        self.variables_n = variables_n
        self.numpoints_x = numpoints_x
        self.numpoints_y = numpoints_y

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.conv1 = nn.Conv2d(5, self.hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.hidden_channels, self.hidden_channels * 2, kernel_size=3, padding=1)
        self.downsample = nn.MaxPool2d(2)
        self.upsample = nn.ConvTranspose2d(self.hidden_channels * 2, self.hidden_channels, kernel_size=2, stride=2)
        self.conv_out = nn.Conv2d(self.hidden_channels, self.variables_n, kernel_size=2, padding=1)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x_params, x_field = x
        x_field = x_field.unsqueeze(1)
        x_params = x_params.view(x_params.size(0), self.parameters_n, 1, 1).expand(-1, -1, self.numpoints_y, self.numpoints_x)
        x_combined = torch.cat([x_field, x_params], dim=1)

        # Forward pass through UNet-like structure
        x = self.conv1(x_combined)
        x = self.downsample(x)
        x = self.conv2(x)
        x = self.upsample(x)
        output = self.conv_out(x)

        return output

class TransformerNet(nn.Module):
    def __init__(self, parameters_n, variables_n, numpoints_x, numpoints_y, **kwargs):
        super(TransformerNet, self).__init__()

        self.parameters_n = parameters_n
        self.variables_n = variables_n
        self.numpoints_x = numpoints_x
        self.numpoints_y = numpoints_y
        self.num_heads = 5

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.embedding = nn.Linear(5, self.hidden_channels)
        self.transformer = nn.Transformer(
            d_model=self.hidden_channels,
            nhead=self.num_heads,
            num_encoder_layers=self.n_layers,
            num_decoder_layers=self.n_layers
        )
        self.fc_out = nn.Linear(self.hidden_channels, self.variables_n)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x_params, x_field = x
        x_field = x_field.unsqueeze(1)
        x_params = x_params.view(x_params.size(0), self.parameters_n, 1, 1).expand(-1, -1, self.numpoints_y, self.numpoints_x)
        x_combined = torch.cat([x_field, x_params], dim=1)

        # Flatten the 2D input into a sequence
        x_flattened = x_combined.view(x_combined.size(0), -1, 5)
        x_embedded = self.embedding(x_flattened)

        # Transformer forward pass
        transformer_output = self.transformer(x_embedded, x_embedded)
        output = self.fc_out(transformer_output)

        # Reshape output back to 2D
        output = output.view(x_combined.size(0), self.variables_n, self.numpoints_y, self.numpoints_x)

        return output

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual  # Add residual connection
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, parameters_n, variables_n, numpoints_x, numpoints_y, **kwargs):
        super(ResNet, self).__init__()

        self.parameters_n = parameters_n
        self.variables_n = variables_n
        self.numpoints_x = numpoints_x
        self.numpoints_y = numpoints_y

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.conv_in = nn.Conv2d(5, self.hidden_channels, kernel_size=3, padding=1)
        self.res_block = ResidualBlock(self.hidden_channels, self.hidden_channels)
        self.conv_out = nn.Conv2d(self.hidden_channels, self.variables_n, kernel_size=1)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x_params, x_field = x
        x_field = x_field.unsqueeze(1)
        x_params = x_params.view(x_params.size(0), self.parameters_n, 1, 1).expand(-1, -1, self.numpoints_y, self.numpoints_x)
        x_combined = torch.cat([x_field, x_params], dim=1)

        # Pass through residual block
        x = self.conv_in(x_combined)
        x = self.res_block(x)
        output = self.conv_out(x)

        return output

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super(ConvLSTMCell, self).__init__()

        self.conv = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, input_tensor, hidden_state):
        h_cur = hidden_state

        combined = torch.cat([input_tensor, h_cur], dim=1)
        h_next = self.conv(combined)

        return h_next

class MLPNet(nn.Module):
    def __init__(self, parameters_n, variables_n, numpoints_x, numpoints_y, **kwargs):
        super(MLPNet, self).__init__()

        self.parameters_n = parameters_n
        self.variables_n = variables_n
        self.numpoints_x = numpoints_x
        self.numpoints_y = numpoints_y

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.fc1 = nn.Linear(self.numpoints_x * self.numpoints_y * 5, self.hidden_channels)
        self.fc2 = nn.Linear(self.hidden_channels, self.hidden_channels)
        self.fc_out = nn.Linear(self.hidden_channels, self.numpoints_x * self.numpoints_y * self.variables_n)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x_params, x_field = x
        x_field = x_field.unsqueeze(1)
        x_params = x_params.view(x_params.size(0), self.parameters_n, 1, 1).expand(-1, -1, self.numpoints_y, self.numpoints_x)
        x_combined = torch.cat([x_field, x_params], dim=1)

        x_flat = x_combined.view(x_combined.size(0), -1)  # Flatten
        x = self.fc1(x_flat)
        x = self.fc2(x)
        output = self.fc_out(x)

        output = output.view(x_combined.size(0), self.variables_n, self.numpoints_y, self.numpoints_x)
        return output

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
        parameters_n,
        variables_n,
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
        Input shape: (batchsize, x=s, y=s, c=parameters_n + 1).
        Output: the solution.
        Output shape: (batchsize, x=s, y=s, c=variables_n).
        """
        self.parameters_n = parameters_n
        self.variables_n = variables_n
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
        x_params = x_params.view(x_params.size(0), self.parameters_n, 1, 1).expand(-1, -1, self.numpoints_y, self.numpoints_x)
        
        # Concatenate parameters and field along the channel dimension
        x_combined = torch.cat([x_field, x_params], dim=1)  # (batch, parameters_n + 1, H, W)
        
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
