from typing import List, Dict, Tuple, Optional, Union

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
        input_vars: List[str],
        output_vars: List[str],
        dataset,
        config,
        spacing: List[float] = [0.03, 0.03],
        g: float = 9.81,
        lambda_physics: float = 0.5,
        epsilon: float = 1e-8
    ):
        super(PhysicsInformedLoss, self).__init__()
        self.g = g
        self.lambda_physics = lambda_physics
        self.data_loss = nn.HuberLoss()
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.dataset = dataset
        self.config = config
        self.spacing = spacing
        self.epsilon = epsilon

    def forward(self, inputs: Tuple[List[torch.Tensor], List[torch.Tensor]], 
                pred: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]], 
                target: Tuple[List[torch.Tensor], List[torch.Tensor]]) -> torch.Tensor:
        """
        Forward method to compute the total loss (data + physics-informed).
        """
        field_target, scalar_target = target
        field_pred, scalar_pred = pred
        
        # Compute data loss
        total_data_loss = 0.0
        
        # Field predictions loss
        if field_pred is not None and field_target:
            field_target_tensor = torch.stack(field_target, dim=1)
            if field_pred.shape != field_target_tensor.shape:
                raise ValueError(f"Shape mismatch: pred {field_pred.shape}, target {field_target_tensor.shape}")
            total_data_loss += self.data_loss(field_pred, field_target_tensor)
        
        # Scalar predictions loss
        if scalar_pred is not None and scalar_target:
            scalar_target_tensor = torch.stack(scalar_target, dim=1)
            total_data_loss += self.data_loss(scalar_pred, scalar_target_tensor)

        # Compute physics loss
        physics_loss, missing_vars = self.compute_physics_loss(inputs, pred)
        
        if missing_vars:
            logging.warning(f"Missing variables for physics loss: {', '.join(missing_vars)}")
            return total_data_loss

        # Combine losses
        total_loss = (1 - self.lambda_physics) * total_data_loss + self.lambda_physics * physics_loss
        return total_loss

    def compute_physics_loss(self, inputs: Tuple[List[torch.Tensor], List[torch.Tensor]], 
                           pred: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]):
        """
        Computes the physics-informed loss using the correct variable assignments.
        """
        # Get variables with correct assignment
        variables, missing_vars = self.assign_variables(inputs, pred)
        if missing_vars:
            return None, missing_vars

        # Extract and process variables
        H = variables["H"]
        U = variables["U"]
        V = variables["V"]
        B = variables["B"]
        n = variables["n"]

        # Ensure proper shapes
        H = H.squeeze(1)
        U = U.squeeze(1)
        V = V.squeeze(1)
        B = B.squeeze(1)
        n = n.permute(1,0,2).expand_as(B)  # Expand n to match spatial dimensions
        
        # Apply minimum value to H for numerical stability
        H = torch.clamp(H, min=self.epsilon)

        # Compute gradients
        dhudx, dhudy = torch.gradient(H * U, spacing=self.spacing, dim=(1, 2))
        dhvdx, dhvdy = torch.gradient(H * V, spacing=self.spacing, dim=(1, 2))
        dhu2dx, dhu2dy = torch.gradient(H * U**2 + 0.5 * self.g * H**2, spacing=self.spacing, dim=(1, 2))
        dhv2dx, dhv2dy = torch.gradient(H * V**2 + 0.5 * self.g * H**2, spacing=self.spacing, dim=(1, 2))
        dhuvdx, dhuvdy = torch.gradient(H * U * V, spacing=self.spacing, dim=(1, 2))
        dzdx, dzdy = torch.gradient(B, spacing=self.spacing, dim=(1, 2))

        # Physics equations
        continuity_loss = dhudx + dhvdy
        
        momentum_x_loss = (
            dhu2dx + dhuvdy 
            - self.g * H * (dzdx - (n**2 * U * torch.sqrt(U**2 + V**2)) / (H**(4/3)))
        )
        
        momentum_y_loss = (
            dhuvdx + dhv2dy 
            - self.g * H * (dzdy - (n**2 * V * torch.sqrt(U**2 + V**2)) / (H**(4/3)))
        )

        # Combine physics losses
        physics_loss = torch.stack([continuity_loss, momentum_x_loss, momentum_y_loss], dim=1)
        return self.data_loss(physics_loss, torch.zeros_like(physics_loss)), []

    def assign_variables(self, inputs: Tuple[List[torch.Tensor], List[torch.Tensor]], 
                        pred: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]):
        """
        Assigns variables following the working reference code pattern.
        """
        variables = {}
        missing_vars = []
        field_inputs, scalar_inputs = inputs
        field_pred, scalar_pred = pred

        # First check if we have all required variables in input_vars and output_vars
        required_vars = {"H", "U", "V", "B", "n"}
        if not required_vars.issubset(set(self.input_vars).union(set(self.output_vars))):
            missing = required_vars - set(self.input_vars).union(set(self.output_vars))
            return None, list(missing)

        # Process input variables
        all_inputs =  (field_inputs if field_inputs is not None else []) + (scalar_inputs if scalar_inputs is not None else [])
        for var, tensor in zip([var for var in self.input_vars if var in self.config.data.non_scalars] + [var for var in self.input_vars if var in self.config.data.scalars], all_inputs):
            if self.config.data.normalize[0]:
                variables[var] = self.dataset._denormalize(tensor, [var])
            else:
                variables[var] = tensor
        
        # Stack tensors into all_preds
        all_preds = torch.cat([field_pred, scalar_pred], dim=1) if field_pred is not None and scalar_pred is not None else field_pred if scalar_pred is None else scalar_pred
        
        
        # Loop through output vars, matching with tensor slices from all_preds
        output_non_scalars = [var for var in self.output_vars if var in self.config.data.non_scalars]
        output_scalars = [var for var in self.output_vars if var in self.config.data.scalars]
        
        # Iterate through vars and tensors together, slicing stacked tensors in all_preds
        for i, (var, tensor_slice) in enumerate(zip(output_non_scalars + output_scalars, torch.unbind(all_preds, dim=1))):
            if self.config.data.normalize[1]:
                variables[var] = self.dataset._denormalize(tensor_slice, [var])
            else:
                variables[var] = tensor_slice
                
        return variables, missing_vars