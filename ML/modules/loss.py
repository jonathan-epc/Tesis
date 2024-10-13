from typing import List

import torch
import torch.nn as nn


class FluidDynamicsLoss(nn.Module):
    def __init__(self, variables: List[str], loss_weights: dict = None):
        super(FluidDynamicsLoss, self).__init__()
        self.variables = variables
        self.mse_loss = nn.MSELoss()
        self.loss_weights = (
            loss_weights
            if loss_weights is not None
            else {var: 1.0 for var in variables}
        )

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor, B: torch.Tensor):
        loss = 0.0
        output_dict = {var: outputs[:, i] for i, var in enumerate(self.variables)}
        target_dict = {var: targets[:, i] for i, var in enumerate(self.variables)}

        for var in self.variables:
            var_loss = self.mse_loss(output_dict[var], target_dict[var])
            loss += self.loss_weights.get(var, 1.0) * var_loss

        if "U" in self.variables and "V" in self.variables:
            U_pred = output_dict["U"]
            V_pred = output_dict["V"]
            velocity_sq = U_pred**2 + V_pred**2

        if all(var in self.variables for var in ["Q", "H", "U", "V"]):
            Q_pred = output_dict["Q"]
            H_pred = output_dict["H"]
            Q_derived = H_pred * torch.sqrt(velocity_sq)
            loss += self.loss_weights.get("Q_penalty", 1.0) * self.mse_loss(
                Q_pred, Q_derived
            )

        if "H" in self.variables and "S" in self.variables:
            H_pred = output_dict["H"]
            S_pred = output_dict["S"]
            H_derived = S_pred - B
            loss += self.loss_weights.get("H_penalty", 1.0) * self.mse_loss(
                H_pred, H_derived
            )

        if all(var in self.variables for var in ["F", "H", "U", "V"]):
            F_pred = output_dict["F"]
            H_pred = output_dict["H"]
            F_derived = torch.sqrt(velocity_sq / (9.81 * torch.clamp(H_pred, min=1e-6)))
            loss += self.loss_weights.get("F_penalty", 1.0) * self.mse_loss(
                F_pred, F_derived
            )

        if torch.isnan(loss).any():
            print("NaN detected in loss computation")
            loss = torch.tensor(
                float("inf")
            )  # Optionally handle it by setting a large loss

        return loss


class PhysicsInformedLoss(nn.Module):
    def __init__(
        self,
        variables: List[str],
        parameters: List[str],
        spacing: List[float] = [0.03, 0.03],
        g=9.81,
        lambda_physics=0.5,
        epsilon=1e-8
    ):
        super(PhysicsInformedLoss, self).__init__()
        self.g = g
        self.lambda_physics = lambda_physics
        self.data_loss = nn.HuberLoss()
        self.variables = variables
        self.parameters = parameters
        self.spacing = spacing
        self.epsilon=epsilon

    def forward(self, pred: torch.Tensor, target: torch.Tensor, params: torch.Tensor):
        # Data loss
        data_loss = self.data_loss(pred, target)

        # Physics loss
        physics_loss = self.compute_physics_loss(pred, params)
        
        # Combine losses
        total_loss = (
            1 - self.lambda_physics
        ) * data_loss + self.lambda_physics * physics_loss

        return total_loss

    def compute_physics_loss(self, pred, params):
        variables = self.assign_arrays_to_variables(pred)
        parameters = self.assign_values_to_parameters(params)

        H = variables["H"]
        H = torch.clamp(H, min=self.epsilon)
        U = variables["U"]
        V = variables["V"]
        n = parameters["n"]
        n = n.unsqueeze(1).unsqueeze(2).expand_as(H)
        z = parameters["B"]

        # Compute gradients
        dhudx, dhudy = torch.gradient(H * U, spacing=self.spacing, dim=(1, 2))
        dhvdx, dhvdy = torch.gradient(H * V, spacing=self.spacing, dim=(1, 2))
        dhuvdx, dhuvdy = torch.gradient(H * U * V, spacing=self.spacing, dim=(1, 2))
        dhu2dx, dhu2dy = torch.gradient(
            H * U**2 + 0.5 * self.g * H**2, spacing=self.spacing, dim=(1, 2)
        )
        dhv2dx, dhv2dy = torch.gradient(
            H * V**2 + 0.5 * self.g * H**2, spacing=self.spacing, dim=(1, 2)
        )
        dzdx, dzdy = torch.gradient(z, spacing=self.spacing, dim=(1, 2))

        # Continuity equation ∂(hu)/∂x + ∂(hv)/∂y = 0
        continuity_loss = dhudx + dhvdy

        # Momentum equation in x-direction (∂(hu^2 + 0.5*gh^2)/∂x) + (∂(huv)/∂y) = gh * ((∂z/∂x) - (n^2 * u * sqrt(u^2 + v^2))/h^(4/3))
        momentum_x_loss = (
            dhu2dx
            + dhuvdy
            - self.g
            * H
            * (dzdx - (n**2 * U * torch.sqrt(U**2 + V**2)) / (H ** (4 / 3)))
        )

        # Momentum equation in y-direction (∂(huv)/∂x) + (∂(hv^2 + 0.5*gh^2)/∂y) = gh * ((∂z/∂y) - (n^2 * v * sqrt(u^2 + v^2))/h^(4/3))
        momentum_y_loss = (
            dhuvdx
            + dhv2dy
            - self.g
            * H
            * (dzdy - (n**2 * V * torch.sqrt(U**2 + V**2)) / (H ** (4 / 3)))
        )

        # Combine all physics losses
        physics_loss = continuity_loss + momentum_x_loss + momentum_y_loss
        physics_loss = self.data_loss(physics_loss, torch.zeros_like(physics_loss))
        return physics_loss

    def assign_arrays_to_variables(self, pred: torch.Tensor):
        if pred.shape[1] != len(self.variables):
            raise ValueError(
                "The number of arrays must match the number of variable names."
            )

        variables = {var_name: pred[:, i] for i, var_name in enumerate(self.variables)}

        return variables

    def assign_values_to_parameters(self, parameters: torch.Tensor):
        if len(parameters) != 2:
            raise ValueError("Expected a list with exactly two elements: params and B.")

        params, B = parameters  # params: [batch_size, num_params], B: [batch_size, ...]

        if "B" not in self.parameters:
            raise ValueError("'B' must be in self.parameters.")

        # Check that the number of params matches the expected number of parameters excluding 'B'
        batch_size, num_params = params.shape
        if num_params != len(self.parameters) - 1:
            raise ValueError(
                f"params length must match the number of parameters excluding 'B'. "
                f"Expected {len(self.parameters) - 1}, got {num_params}."
            )

        # Assign B for each batch
        parameters_dict = {"B": B}

        # Assign the other parameters, excluding B
        param_names = [p for p in self.parameters if p != "B"]

        # Now zip param names and params (batch-wise)
        for i, param_name in enumerate(param_names):
            parameters_dict[param_name] = params[:, i]

        return parameters_dict