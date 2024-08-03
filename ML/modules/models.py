import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.models import FNO, SFNO, UNO
from typing import Tuple

class FNOnet(nn.Module):
    """
    Fourier Neural Operator Network (FNOnet) for processing 2D fields with parameter embedding.

    This class implements a neural network that combines parameter embedding and
    a Fourier Neural Operator (FNO) to process 2D fields with associated parameters.

    Attributes:
        parameters_n (int): Number of input parameters.
        variables_n (int): Number of output variables.
        numpoints_x (int): Number of points in the x-dimension of the field.
        numpoints_y (int): Number of points in the y-dimension of the field.
        n_modes (Tuple[int, int]): Number of Fourier modes in each dimension.
        hidden_channels (int): Number of hidden channels in the FNO.
        n_layers (int): Number of FNO layers.
        lifting_channels (int): Number of lifting channels.
        projection_channels (int): Number of projection channels.
        param_embedding (nn.Sequential): Neural network for parameter embedding.
        fno (FNO): Fourier Neural Operator module.

    """

    def __init__(
        self,
        parameters_n: int,
        variables_n: int,
        numpoints_x: int,
        numpoints_y: int,
        n_modes: Tuple[int, int] = (4, 200),
        hidden_channels: int = 11,
        n_layers: int = 5,
        lifting_channels: int = 16,
        projection_channels: int = 16,
    ):
        """
        Initialize the FNOnet.

        Args:
            parameters_n (int): Number of input parameters.
            variables_n (int): Number of output variables.
            numpoints_x (int): Number of points in the x-dimension of the field.
            numpoints_y (int): Number of points in the y-dimension of the field.
            n_modes (Tuple[int, int], optional): Number of Fourier modes in each dimension. Defaults to (4, 200).
            hidden_channels (int, optional): Number of hidden channels in the FNO. Defaults to 11.
            n_layers (int, optional): Number of FNO layers. Defaults to 5.
            lifting_channels (int, optional): Number of lifting channels. Defaults to 16.
            projection_channels (int, optional): Number of projection channels. Defaults to 16.
        """
        super(FNOnet, self).__init__()
        self.parameters_n = parameters_n
        self.variables_n = variables_n
        self.numpoints_x = numpoints_x
        self.numpoints_y = numpoints_y

        # Embedding layer for parameters
        self.param_embedding = nn.Sequential(
            nn.Linear(parameters_n, 64),
            nn.ReLU(),
            nn.Linear(64, numpoints_y * numpoints_x),
        )

        # FNO for 2D field
        self.fno = FNO(
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            in_channels=2,  # 1 for the field channel and 1 for the embedded parameters
            out_channels=variables_n,
            n_layers=n_layers,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
        )

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the FNOnet.

        Args:
            x (Tuple[torch.Tensor, torch.Tensor]): Input tuple containing:
                - x_params (torch.Tensor): Input parameters tensor of shape (batch_size, parameters_n).
                - x_field (torch.Tensor): Input field tensor of shape (batch_size, numpoints_y, numpoints_x).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, variables_n, numpoints_y, numpoints_x).
        """
        x_params, x_field = x

        # Embed and reshape parameters to match field size
        x_params = self.param_embedding(x_params).view(
            -1, 1, self.numpoints_y, self.numpoints_x
        )

        # Concatenate parameters and field along channel dimension
        x_combined = torch.cat([x_field.unsqueeze(1), x_params], dim=1)

        # Forward pass through FNO
        output = self.fno(x_combined)

        return output

    def __repr__(self):
        """
        Return a string representation of the FNOnet.

        Returns:
            str: String representation of the FNOnet.
        """
        return (f"FNOnet(parameters_n={self.parameters_n}, "
                f"variables_n={self.variables_n}, "
                f"numpoints_x={self.numpoints_x}, "
                f"numpoints_y={self.numpoints_y})")

    @property
    def device(self) -> torch.device:
        """
        Get the device on which the model parameters are stored.

        Returns:
            torch.device: The device (CPU or GPU) of the model parameters.
        """
        return next(self.parameters()).device

    def to(self, device: torch.device):
        """
        Move the model to the specified device.

        Args:
            device (torch.device): The target device (CPU or GPU).

        Returns:
            FNOnet: The model instance moved to the specified device.
        """
        return super().to(device)

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


class FNO2d(nn.Module):
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
    ):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """
        self.parameters_n = parameters_n
        self.numpoints_x = numpoints_x
        self.numpoints_y = numpoints_y

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9  # pad the domain if input is non-periodic
        self.input_channels = 1 + parameters_n + 2
        self.output_channels = output_channels

        self.p = nn.Linear(
            self.input_channels, self.width
        )  # input channel is 3: (a(x, y), x, y)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.q = MLP(
            self.width, self.output_channels, self.width * 4
        )  # output channel is 1: u(x, y)

    def forward(self, x):
        x_params, x_field = x
        x_params = (
            x_params.unsqueeze(1)
            .unsqueeze(2)
            .expand(-1, self.numpoints_y, self.numpoints_x, -1)
        )
        x = torch.cat([x_field.unsqueeze(-1), x_params], dim=-1)

        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0, self.padding, 0, self.padding])

        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., : -self.padding, : -self.padding]
        x = self.q(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)