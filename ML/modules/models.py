import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.models import FNO, SFNO, UNO


class FNOnet(nn.Module):
    """
    FNOnet2 is a neural network model that combines fully connected layers for parameters
    with a Fourier Neural Operator (FNO) for 2D field processing.

    Args:
        parameters_n (int): Number of parameter inputs.
        variables_n (int): Number of output variables.
        numpoints_x (int): Number of points in the x dimension of the field.
        numpoints_y (int): Number of points in the y dimension of the field.
        n_modes (tuple, optional): Number of Fourier modes in each dimension.
        hidden_channels (int, optional): Number of hidden channels in the FNO.
        n_layers (int, optional): Number of layers in the FNO.
        lifting_channels (int, optional): Number of lifting channels in the FNO.
        projection_channels (int, optional): Number of projection channels in the FNO.
    """

    def __init__(
        self,
        parameters_n,
        variables_n,
        numpoints_x,
        numpoints_y,
        n_modes=(4, 200),
        hidden_channels=11,
        n_layers=5,
        lifting_channels=16,
        projection_channels=16,
    ):
        super(FNOnet, self).__init__()

        self.parameters_n = parameters_n
        self.numpoints_x = numpoints_x
        self.numpoints_y = numpoints_y

        self.fno = FNO(
            n_modes=n_modes,  # Number of Fourier modes in each dimension
            hidden_channels=hidden_channels,  # Number of hidden channels in the FNO
            in_channels=parameters_n
            + 1,  # Adjusted to include parameter channels and the field channel
            out_channels=variables_n,  # Number of output variables
            n_layers=n_layers,  # Number of FNO layers
            lifting_channels=lifting_channels,  # Number of lifting channels
            projection_channels=projection_channels,  # Number of projection channels
        )

    def forward(self, x):
        x_params, x_field = x

        # Reshape parameters to match field size
        x_params = (
            x_params.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, self.parameters_n, self.numpoints_y, self.numpoints_x)
        )
        # Concatenate parameters and field along channel dimension
        x_field = torch.cat([x_field.unsqueeze(1), x_params], dim=1)

        # Forward pass for 2D field branch using SFNO
        x_field = self.fno(x_field)

        # Flatten output
        x_field = x_field.view(x_field.size(0), -1)

        return x_field


class Simplenet(nn.Module):
    def __init__(self, parameters_n, variables_n, numpoints_x, numpoints_y):
        super(Simplenet, self).__init__()

        self.parameters_n = parameters_n
        self.numpoints_x = numpoints_x
        self.numpoints_y = numpoints_y

        self.linear1 = nn.Sequential(
            nn.Linear(parameters_n + numpoints_y * numpoints_x, 64),
            nn.ReLU(),
            nn.Linear(64, numpoints_y * numpoints_x * variables_n),
        )

    def forward(self, x):
        x_params, x_field = x
        x_params = x_params.view(x_params.size(0), -1)
        x_field = x_field.view(x_field.size(0), -1)
        x = torch.cat([x_params, x_field], dim=1)
        x = self.linear1(x)

        return x


class FNOv2net(nn.Module):
    def __init__(
        self,
        parameters_n,
        variables_n,
        numpoints_x,
        numpoints_y,
        n_modes=(4, 200),
        hidden_channels=11,
        n_layers=5,
        lifting_channels=16,
        projection_channels=16,
    ):
        super(FNOv2net, self).__init__()

        self.parameters_n = parameters_n
        self.numpoints_x = numpoints_x
        self.numpoints_y = numpoints_y

        # Embedding layer for parameters
        self.param_embedding = nn.Sequential(
            nn.Linear(parameters_n, 64),
            nn.ReLU(),
            nn.Linear(64, numpoints_y * numpoints_x),
        )

        # Embedding layer for parameters
        self.param_embedding = nn.Sequential(
            nn.Linear(parameters_n, 64),
            nn.ReLU(),
            nn.Linear(64, numpoints_y * numpoints_x),
        )

        # Branch 2: FNO for 2D field
        self.fno = FNO(
            n_modes=n_modes,  # Number of Fourier modes in each dimension
            hidden_channels=hidden_channels,  # Number of hidden channels in the FNO
            in_channels=2,  # Adjusted to include parameter channels and the field channel
            out_channels=variables_n,  # Number of output variables
            n_layers=n_layers,  # Number of FNO layers
            lifting_channels=lifting_channels,  # Number of lifting channels
            projection_channels=projection_channels,  # Number of projection channels
        )

    def forward(self, x):
        x_params, x_field = x
        # Embed and reshape parameters to match field size
        x_params = self.param_embedding(x_params).view(
            -1, 1, self.numpoints_y, self.numpoints_x
        )
        # Concatenate parameters and field along channel dimension
        x_field = torch.cat([x_field.unsqueeze(1), x_params], dim=1)

        # Forward pass for 2D field branch using SFNO
        x_field = self.fno(x_field)

        return x_field


class FNOv3net(nn.Module):
    def __init__(
        self,
        parameters_n,
        variables_n,
        numpoints_x,
        numpoints_y,
        n_modes=(4, 200),
        hidden_channels=11,
        n_layers=5,
        lifting_channels=16,
        projection_channels=16,
    ):
        super(FNOv3net, self).__init__()

        self.parameters_n = parameters_n
        self.numpoints_x = numpoints_x
        self.numpoints_y = numpoints_y

        # Embedding layer for parameters
        self.param_embedding = nn.Sequential(
            nn.Linear(parameters_n, 64),
            nn.ReLU(),
            nn.Linear(64, numpoints_y * numpoints_x),
        )
        self.mlp = nn.Sequential(
            nn.Conv2d(variables_n, 2 * variables_n, 1),
            nn.ReLU(),
            nn.Conv2d(2 * variables_n, variables_n, 1),
        )
        # Branch 2: SFNO for 2D field
        self.fno = FNO(
            n_modes=n_modes,  # Number of Fourier modes in each dimension
            hidden_channels=hidden_channels,  # Number of hidden channels in the FNO
            in_channels=1
            + 1
            + 2,  # Adjusted to include parameter channels and the field channel
            out_channels=variables_n,  # Number of output variables
            n_layers=n_layers,  # Number of FNO layers
            lifting_channels=lifting_channels,  # Number of lifting channels
            projection_channels=projection_channels,  # Number of projection channels
        )
        # Generate coordinate grids
        self.register_buffer(
            "x_grid", torch.linspace(0, 12, numpoints_x).view(1, 1, 1, numpoints_x)
        )
        self.register_buffer(
            "y_grid", torch.linspace(0, 0.3, numpoints_y).view(1, 1, numpoints_y, 1)
        )

    def forward(self, x):
        x_params, x_field = x
        # Ae = x_params.expand(self.numpoints_y,self.numpoints_x,-1,self.parameters_n).permute(2,3,0,1)
        Ae = (
            self.param_embedding(x_params)
            .view(-1, self.numpoints_y, self.numpoints_x)
            .unsqueeze(1)
        )
        Be = x_field.unsqueeze(1)
        Ce = self.x_grid.expand(Ae.size(0), 1, self.numpoints_y, self.numpoints_x)
        De = self.y_grid.expand(Ae.size(0), 1, self.numpoints_y, self.numpoints_x)
        x_field = torch.cat([Ae, Be, Ce, De], dim=1)
        x_field = self.fno(x_field)
        x_field = self.mlp(x_field)

        return x_field


class SFNOnet(nn.Module):
    """
    FNOnet2 is a neural network model that combines fully connected layers for parameters
    with a Fourier Neural Operator (FNO) for 2D field processing.

    Args:
        parameters_n (int): Number of parameter inputs.
        variables_n (int): Number of output variables.
        numpoints_x (int): Number of points in the x dimension of the field.
        numpoints_y (int): Number of points in the y dimension of the field.
        n_modes (tuple, optional): Number of Fourier modes in each dimension.
        hidden_channels (int, optional): Number of hidden channels in the FNO.
        n_layers (int, optional): Number of layers in the FNO.
        lifting_channels (int, optional): Number of lifting channels in the FNO.
        projection_channels (int, optional): Number of projection channels in the FNO.
    """

    def __init__(
        self,
        parameters_n,
        variables_n,
        numpoints_x,
        numpoints_y,
        n_modes=(4, 200),
        hidden_channels=11,
        n_layers=5,
        lifting_channels=16,
        projection_channels=16,
    ):
        super(SFNOnet, self).__init__()

        self.parameters_n = parameters_n
        self.numpoints_x = numpoints_x
        self.numpoints_y = numpoints_y

        self.fno = SFNO(
            n_modes=n_modes,  # Number of Fourier modes in each dimension
            hidden_channels=hidden_channels,  # Number of hidden channels in the FNO
            in_channels=parameters_n
            + 1,  # Adjusted to include parameter channels and the field channel
            out_channels=variables_n,  # Number of output variables
            n_layers=n_layers,  # Number of FNO layers
            lifting_channels=lifting_channels,  # Number of lifting channels
            projection_channels=projection_channels,  # Number of projection channels
            factorization="dense",
        )

    def forward(self, x):
        x_params, x_field = x

        # Reshape parameters to match field size
        x_params = (
            x_params.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, self.parameters_n, self.numpoints_y, self.numpoints_x)
        )
        # Concatenate parameters and field along channel dimension
        x_field = torch.cat([x_field.unsqueeze(1), x_params], dim=1)

        # Forward pass for 2D field branch using SFNO
        x_field = self.fno(x_field)

        # Flatten output
        x_field = x_field.view(x_field.size(0), -1)

        return x_field


class UNOnet(nn.Module):
    """
    FNOnet2 is a neural network model that combines fully connected layers for parameters
    with a Fourier Neural Operator (FNO) for 2D field processing.

    Args:
        parameters_n (int): Number of parameter inputs.
        variables_n (int): Number of output variables.
        numpoints_x (int): Number of points in the x dimension of the field.
        numpoints_y (int): Number of points in the y dimension of the field.
        n_modes (tuple, optional): Number of Fourier modes in each dimension.
        hidden_channels (int, optional): Number of hidden channels in the FNO.
        n_layers (int, optional): Number of layers in the FNO.
        lifting_channels (int, optional): Number of lifting channels in the FNO.
        projection_channels (int, optional): Number of projection channels in the FNO.
    """

    def __init__(
        self,
        parameters_n,
        variables_n,
        numpoints_x,
        numpoints_y,
        n_modes=(4, 200),
        hidden_channels=11,
        n_layers=5,
        lifting_channels=16,
        projection_channels=16,
    ):
        super(UNOnet, self).__init__()

        self.parameters_n = parameters_n
        self.numpoints_x = numpoints_x
        self.numpoints_y = numpoints_y

        self.uno = UNO(
            n_modes=n_modes,  # Number of Fourier modes in each dimension
            hidden_channels=hidden_channels,  # Number of hidden channels in the FNO
            in_channels=parameters_n
            + 1,  # Adjusted to include parameter channels and the field channel
            out_channels=variables_n,  # Number of output variables
            n_layers=5,  # Number of FNO layers
            lifting_channels=lifting_channels,  # Number of lifting channels
            projection_channels=projection_channels,  # Number of projection channels
            uno_out_channels=[16, 32, 64, 32, 16],
            uno_n_modes=[[8, 8], [16, 16], [32, 32], [16, 16], [8, 8]],
            uno_scalings=[[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]],
        )

    def forward(self, x):
        x_params, x_field = x

        # Reshape parameters to match field size
        x_params = (
            x_params.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, self.parameters_n, self.numpoints_y, self.numpoints_x)
        )
        # Concatenate parameters and field along channel dimension
        x_field = torch.cat([x_field.unsqueeze(1), x_params], dim=1)

        # Forward pass for 2D field branch using SFNO
        x_field = self.uno(x_field)

        # Flatten output
        x_field = x_field.view(x_field.size(0), -1)

        return x_field


class ImprovedFNOnet(nn.Module):
    def __init__(self, parameters_n, variables_n, numpoints_x, numpoints_y):
        super(ImprovedFNOnet, self).__init__()
        self.parameters_n = parameters_n
        self.variables_n = variables_n
        self.numpoints_x = numpoints_x
        self.numpoints_y = numpoints_y

        # Multi-scale FNO
        self.fno_layers = nn.ModuleList(
            [
                FNO(
                    n_modes=(2, 2),
                    hidden_channels=16,
                    in_channels=parameters_n + 1,
                    out_channels=16,
                ),
                FNO(
                    n_modes=(4, 4), hidden_channels=32, in_channels=16, out_channels=32
                ),
                FNO(
                    n_modes=(8, 8), hidden_channels=64, in_channels=32, out_channels=64
                ),
            ]
        )

        # Projection layers for residual connections
        self.projections = nn.ModuleList(
            [
                nn.Conv2d(parameters_n + 1, 16, 1),
                nn.Conv2d(16, 32, 1),
                nn.Conv2d(32, 64, 1),
            ]
        )

        # Output heads
        self.output_heads = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(64, 32, 1), nn.ReLU(), nn.Conv2d(32, 1, 1))
                for _ in range(variables_n)
            ]
        )

    def forward(self, x):
        x_params, x_field = x

        # Reshape parameters to match field size
        x_params = (
            x_params.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, self.numpoints_y, self.numpoints_x)
        )

        # Concatenate parameters and field along channel dimension
        x = torch.cat([x_field.unsqueeze(1), x_params], dim=1)

        # Multi-scale FNO with residual connections
        for fno_layer, projection in zip(self.fno_layers, self.projections):
            residual = projection(x)
            x = fno_layer(x) + residual

        # Separate output heads for each variable
        outputs = [head(x) for head in self.output_heads]
        outputs = torch.cat(outputs, dim=1)

        return outputs

    def compute_loss(self, predictions, targets):
        # Simple MSE loss
        return F.mse_loss(predictions, targets)


class UNet(nn.Module):
    def __init__(
        self,
        parameters_n,
        variables_n,
        numpoints_x,
        numpoints_y,
        in_channels=5,
        out_channels=6,
    ):
        super(UNet, self).__init__()
        self.parameters_n = parameters_n
        self.numpoints_x = numpoints_x
        self.numpoints_y = numpoints_y

        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        self.middle = self.conv_block(512, 1024)

        self.decoder4 = self.conv_block(1024 + 512, 512)
        self.decoder3 = self.conv_block(512 + 256, 256)
        self.decoder2 = self.conv_block(256 + 128, 128)
        self.decoder1 = self.conv_block(128 + 64, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x_params, x_field = x
        x_params = (
            x_params.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, self.numpoints_y, self.numpoints_x)
        )
        x = torch.cat([x_field.unsqueeze(1), x_params], dim=1)

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))
        enc4 = self.encoder4(F.max_pool2d(enc3, 2))

        middle = self.middle(F.max_pool2d(enc4, 2))

        dec4 = self.decoder4(
            torch.cat([F.interpolate(middle, scale_factor=2), enc4], dim=1)
        )
        dec3 = self.decoder3(
            torch.cat([F.interpolate(dec4, scale_factor=2), enc3], dim=1)
        )
        dec2 = self.decoder2(
            torch.cat([F.interpolate(dec3, scale_factor=2), enc2], dim=1)
        )
        dec1 = self.decoder1(
            torch.cat([F.interpolate(dec2, scale_factor=2), enc1], dim=1)
        )

        out = self.final(dec1)

        return out.view(out.size(0), -1)


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