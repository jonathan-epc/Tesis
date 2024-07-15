import torch
import torch.nn as nn
from neuralop.models import FNO


class FNOnet2(nn.Module):
    def __init__(self, parameters_n, variables_n, numpoints_x, numpoints_y):
        super(FNOnet2, self).__init__()

        # Branch 1: Fully Connected layers for parameters
        self.parameters_n = parameters_n
        self.numpoints_x = numpoints_x
        self.numpoints_y = numpoints_y

        # Branch 2: SFNO for 2D field
        self.fno = FNO(
            n_modes=(4, 4),  # Adjust based on your data
            hidden_channels=16,
            in_channels=parameters_n + 1,  # Changed here
            out_channels=variables_n,  # Changed here
            n_layers=4,
            lifting_channels=16,
            projection_channels=16,
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


class FNOnet(nn.Module):
    def __init__(self, parameters_n, variables_n, numpoints_x, numpoints_y):
        super(FNOnet, self).__init__()

        # Branch 1: Fully Connected layers for parameters
        self.fc1_params = nn.Linear(parameters_n, 16)
        self.fc2_params = nn.Linear(16, parameters_n)

        # Branch 2: SFNO for 2D field
        self.fno = FNO(
            n_modes=(16, 16),  # Adjust based on your data
            hidden_channels=16,
            in_channels=1,
            out_channels=6,
            n_layers=4,
            lifting_channels=16,
            projection_channels=16,
        )

        # Fully Connected layers for the combined output
        self.fc1_combined = nn.Linear(parameters_n + 6 * numpoints_x * numpoints_y, 32)
        self.fc2_combined = nn.Linear(32, variables_n * numpoints_x * numpoints_y)

    def forward(self, x):
        x_params, x_field = x

        # Forward pass for parameters branch
        x_params = F.relu(self.fc1_params(x_params))
        x_params = F.relu(self.fc2_params(x_params))

        # Forward pass for 2D field branch using SFNO
        x_field = x_field.unsqueeze(1)  # Add a channel dimension
        x_field = self.fno(x_field)
        x_field = x_field.view(x_field.size(0), -1)  # Flatten

        # Combine both branches
        x_combined = torch.cat([x_params, x_field], dim=1)
        x_combined = F.relu(self.fc1_combined(x_combined))
        x_combined = self.fc2_combined(x_combined)

        return x_combined


class ComplexSFNO(nn.Module):
    def __init__(self, parameters_n, variables_n, numpoints_x, numpoints_y):
        super(ComplexSFNO, self).__init__()

        # Branch 1: Fully Connected layers for parameters
        self.fc1_params = nn.Linear(parameters_n, 16)
        self.fc2_params = nn.Linear(16, 32)

        # Branch 2: SFNO for 2D field
        self.sfno = SFNO(
            n_modes=(16, 16),  # Adjust based on your data
            hidden_channels=16,
            in_channels=1,
            out_channels=8,
            n_layers=4,
            lifting_channels=16,
            projection_channels=16,
        )

        # Fully Connected layers for the combined output
        self.fc1_combined = nn.Linear(32 + 8 * numpoints_x * numpoints_y, 32)
        self.fc2_combined = nn.Linear(32, variables_n * numpoints_x * numpoints_y)

        # Batch normalization layers
        self.batch_norm_fc = nn.BatchNorm1d(32)
        self.batch_norm_combined = nn.BatchNorm1d(32)

    def forward(self, x):
        x_params, x_field = x

        # Forward pass for parameters branch
        x_params = F.relu(self.fc1_params(x_params))
        x_params = F.relu(self.batch_norm_fc(self.fc2_params(x_params)))

        # Forward pass for 2D field branch using SFNO
        x_field = x_field.unsqueeze(1)  # Add a channel dimension
        x_field = self.sfno(x_field)
        x_field = x_field.view(x_field.size(0), -1)  # Flatten

        # Combine both branches
        x_combined = torch.cat([x_params, x_field], dim=1)
        x_combined = F.relu(self.batch_norm_combined(self.fc1_combined(x_combined)))
        x_combined = self.fc2_combined(x_combined)

        return x_combined


class ComplexSFNO(nn.Module):
    def __init__(self, parameters_n, variables_n, numpoints_x, numpoints_y):
        super(ComplexSFNO, self).__init__()

        # Branch 2: SFNO for 2D field
        self.sfno = SFNO(
            n_modes=(16, 16),  # Adjust based on your data
            hidden_channels=16,
            in_channels=1,
            out_channels=8,
            n_layers=4,
            lifting_channels=16,
            projection_channels=16,
        )

        # Fully Connected layers for the combined output
        self.fc1 = nn.Linear(8 * numpoints_x * numpoints_y, 32)
        self.fc2 = nn.Linear(32, variables_n * numpoints_x * numpoints_y)

        # Batch normalization layers
        self.batch_norm = nn.BatchNorm1d(32)

    def forward(self, x):
        x_params, x_field = x

        # Forward pass for 2D field branch using SFNO
        x_field = x_field.unsqueeze(1)  # Add a channel dimension
        x_field = self.sfno(x_field)

        # Flatten the output of SFNO
        x_field = x_field.view(x_field.size(0), -1)

        # Pass through fully connected layers
        x_field = self.fc1(x_field)
        x_field = self.batch_norm(x_field)
        x_field = F.relu(x_field)
        x_field = self.fc2(x_field)

        return x_field


class ComplexNN(nn.Module):
    def __init__(self, parameters_n, variables_n, numpoints_x, numpoints_y):
        super(ComplexNN, self).__init__()

        # Branch 1: Fully Connected layers for parameters
        self.fc1_params = nn.Linear(parameters_n, 64)
        self.fc2_params = nn.Linear(64, 128)

        # Branch 2: Convolutional layers for 2D field
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected layers for the combined output
        self.fc1_combined = nn.Linear(
            64 * (numpoints_x // 2 // 2) * (numpoints_y // 2 // 2) + 128, 256
        )
        self.fc2_combined = nn.Linear(256, variables_n * numpoints_x * numpoints_y)

        # Batch normalization layers
        self.batch_norm_fc = nn.BatchNorm1d(128)
        self.batch_norm_conv = nn.BatchNorm2d(64)
        self.batch_norm_combined = nn.BatchNorm1d(256)

    def forward(self, x):
        x_params, x_field = x

        # Forward pass for parameters branch
        x_params = F.relu(self.fc1_params(x_params))
        x_params = F.relu(self.batch_norm_fc(self.fc2_params(x_params)))

        # Forward pass for 2D field branch
        x_field = x_field.unsqueeze(1)  # Add a channel dimension
        x_field = F.relu(self.conv1(x_field))
        x_field = self.pool(F.relu(self.conv2(x_field)))
        x_field = self.pool(F.relu(self.batch_norm_conv(self.conv3(x_field))))
        x_field = x_field.view(x_field.size(0), -1)  # Flatten

        # Combine both branches
        x_combined = torch.cat([x_params, x_field], dim=1)
        x_combined = F.relu(self.batch_norm_combined(self.fc1_combined(x_combined)))
        x_combined = self.fc2_combined(x_combined)

        return x_combined


class SimpleNN(nn.Module):

    def __init__(self, parameters_n, variables_n, numpoints_x, numpoints_y):
        super(SimpleNN, self).__init__()
        self.conv1 = nn.Conv2d(1, variables_n, 3)
        self.fc1 = nn.Linear(
            6 * (numpoints_x - 3 + 1) * (numpoints_y - 3 + 1) + parameters_n, 32
        )
        self.fc2 = nn.Linear(32, variables_n * numpoints_x * numpoints_y)

    def forward(self, x):
        x0, x1 = x
        x1 = self.conv1(x1.unsqueeze(1))
        x1 = F.relu(x1)
        x1 = x1.view(x1.size(0), -1)
        x = torch.cat([x0, x1], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class SimpleNN(nn.Module):

    def __init__(self, parameters_n, variables_n, numpoints_x, numpoints_y):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(parameters_n + numpoints_x * numpoints_y, 2)
        self.fc2 = nn.Linear(2, variables_n * numpoints_x * numpoints_y)
        self.batch_norm1 = nn.BatchNorm1d(2)

    def forward(self, x):
        x = torch.cat([x[0], x[1].view(x[1].size(0), -1)], 1)
        x = F.relu(self.batch_norm1(self.fc1(x)))
        x = self.fc2(x)
        return x