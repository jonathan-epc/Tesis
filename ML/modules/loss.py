from typing import List, Dict, Tuple, Optional, Union

import torch
import torch.nn as nn
from loguru import logger
import random


class PhysicsInformedLoss(nn.Module):
    def __init__(
        self,
        input_vars: List[str],
        output_vars: List[str],
        config,
        dataset,
        spacing: List[float] = [0.03, 0.03],
        g: float = 9.81,
        alpha: float = 0.999,
        temperature: float = 0.1,
        rho: float = 0.99,
        lambda_physics: float = 0.5,
        epsilon: float = 1e-5
    ):
        super(PhysicsInformedLoss, self).__init__()
        self.g = g
        self.lambda_physics = lambda_physics
        self.data_loss = nn.HuberLoss()
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.config = config
        self.dataset = dataset
        self.epsilon = epsilon
        self.alpha = alpha
        self.temperature = temperature
        self.rho = rho
        self.call_count = 0
        if self.config.data.adimensional:
            self.spacing = spacing[0] / self.config.channel.length, spacing[1] / self.config.channel.width
        else:
            self.spacing = spacing
        # Initialize ReLoBRaLo variables
        self.lambdas = torch.ones(4)  # Two loss terms: data and physics
        self.last_losses = torch.ones(4)
        self.init_losses = torch.ones(4)

    def forward(
        self,
        inputs: Tuple[List[torch.Tensor], List[torch.Tensor]],
        pred: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]],
        target: Tuple[List[torch.Tensor], List[torch.Tensor]],
    ) -> torch.Tensor:
        field_target, scalar_target = target
        field_pred, scalar_pred = pred
        
        # Compute data loss
        total_data_loss = 0.0
        if field_pred is not None and field_target:
            field_target_tensor = torch.stack(field_target, dim=1)
            if field_pred.shape != field_target_tensor.shape:
                raise ValueError(f"Shape mismatch: field_pred {field_pred.shape}, field_target {field_target_tensor.shape}")
            total_data_loss += self.data_loss(field_pred, field_target_tensor)
        if scalar_pred is not None and scalar_target:
            scalar_target_tensor = torch.stack(scalar_target, dim=1)
            if scalar_pred.shape != scalar_target_tensor.shape:
                raise ValueError(f"Shape mismatch: scalar_pred {scalar_pred.shape}, scalar_target {scalar_target_tensor.shape}")
            total_data_loss += self.data_loss(scalar_pred, scalar_target_tensor)

        
        # Check if physics loss should be computed
        zero_tensor = torch.tensor(0.0, device=total_data_loss.device)
        if not hasattr(self.config.training, 'use_physics_loss') or not self.config.training.use_physics_loss:
            return total_data_loss, total_data_loss, zero_tensor, zero_tensor, zero_tensor
        
        # Compute physics loss
        compute_physics = self.compute_adimensional_physics_loss if self.config.data.adimensional else self.compute_physics_loss
        physics_loss, missing_vars = compute_physics(inputs, pred)
        
        if missing_vars:
            logger.warning(f"Missing variables for physics loss: {', '.join(missing_vars)}")
            return total_data_loss, total_data_loss, zero_tensor, zero_tensor, zero_tensor
        
        continuity_loss, momentum_x_loss, momentum_y_loss = physics_loss
        
        # Update ReLoBRaLo lambdas
        self.update_relobralo_lambdas(total_data_loss, continuity_loss, momentum_x_loss, momentum_y_loss)

        
        # Combine losses with dynamic weighting
        total_loss = (
            self.lambdas[0] * total_data_loss + 
            self.lambdas[1] * continuity_loss + 
            self.lambdas[2] * momentum_x_loss + 
            self.lambdas[3] * momentum_y_loss
        )
        
        return total_loss, total_data_loss, continuity_loss, momentum_x_loss, momentum_y_loss

    def update_relobralo_lambdas(self, total_data_loss, continuity_loss, momentum_x_loss, momentum_y_loss):
        losses = torch.tensor([total_data_loss.item(), continuity_loss.item(), momentum_x_loss.item(), momentum_y_loss.item()])

        # Compute alpha and rho based on iteration
        alpha = 1.0 if self.call_count == 0 else (0.0 if self.call_count == 1 else self.alpha)
        rho = 1.0 if self.call_count <= 1 else torch.bernoulli(torch.tensor(self.rho)).item()

        # Compute lambdas_hat (current iteration)
        lambdas_hat = losses / (self.last_losses * self.temperature + self.epsilon)
        lambdas_hat = torch.softmax(lambdas_hat - torch.max(lambdas_hat), dim=0) * len(losses)

        # Compute init_lambdas_hat (random lookback)
        init_lambdas_hat = losses / (self.init_losses * self.temperature + self.epsilon)
        init_lambdas_hat = torch.softmax(init_lambdas_hat - torch.max(init_lambdas_hat), dim=0) * len(losses)

        # Update lambdas
        new_lambdas = (
            rho * alpha * self.lambdas
            + (1 - rho) * alpha * init_lambdas_hat
            + (1 - alpha) * lambdas_hat
        )
        self.lambdas = new_lambdas.detach()

        # Update loss history
        self.last_losses = losses.detach()
        if self.call_count == 0:
            self.init_losses = losses.detach()

        self.call_count += 1

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
        h = variables["H"]
        u = variables["U"]
        v = variables["V"]
        b = variables["B"]
        n = variables["n"]
        nut = variables["nut"]
        n = n.view(-1, 1, 1) 
        nut = nut.view(-1, 1, 1) 
       
        # Apply minimum value to H for numerical stability, and to values that can't be negative
        h = torch.clamp(h, min=self.epsilon)
        b = torch.clamp(b, min=0)
        n = torch.clamp(n, min=0)
        nut = torch.clamp(nut, min=0)
        absU = torch.sqrt(torch.clamp(u**2 + v**2, min=self.epsilon))

        # Compute gradients for continuity loss       
        d_hu_dx, d_hu_dy = torch.gradient(h * u, spacing=self.spacing, dim=(1, 2))
        d_hv_dx, d_hv_dy = torch.gradient(h * v, spacing=self.spacing, dim=(1, 2))
        d_h_dx, d_h_dy = torch.gradient(h, spacing=self.spacing, dim=(1, 2))
        d_u_dx, d_u_dy = torch.gradient(u, spacing=self.spacing, dim=(1, 2))
        d_v_dx, d_v_dy = torch.gradient(v, spacing=self.spacing, dim=(1, 2))
        d_z_dx, d_z_dy = torch.gradient(h+b, spacing=self.spacing, dim=(1, 2))
        d_u_dx2, d_u_dxdy = torch.gradient(d_u_dx, spacing=self.spacing, dim=(1, 2))
        d_v_dx2, d_v_dxdy = torch.gradient(d_v_dx, spacing=self.spacing, dim=(1, 2))
        d_u_dydx, d_u_dy2 = torch.gradient(d_u_dy, spacing=self.spacing, dim=(1, 2))
        d_v_dydx, d_v_dy2 = torch.gradient(d_v_dy, spacing=self.spacing, dim=(1, 2))

        continuity_loss = d_hu_dx + d_hv_dy

        advection_x = u * d_u_dx + v * d_u_dy
        advection_y = u * d_v_dx + v * d_v_dy

        pressure_x = -self.g * d_z_dx
        pressure_y = -self.g * d_z_dy

        friction_x = -self.g * (n ** 2) / (h ** (4/3)) * absU * u
        friction_y = -self.g * (n ** 2) / (h ** (4/3)) * absU * v

        diffusion_x = nut/h * (d_h_dx * d_u_dx + d_h_dy * d_u_dy + h * (d_u_dx2 + d_u_dy2))
        diffusion_y = nut/h * (d_h_dx * d_v_dx + d_h_dy * d_v_dy + h * (d_v_dx2 + d_v_dy2))

        momentum_x_loss = h ** (4/3) * (advection_x - (pressure_x + friction_x + diffusion_x))
        momentum_y_loss = h ** (4/3) * (advection_y - (pressure_y + friction_y + diffusion_y))

        # Combine physics losses
        # physics_loss = torch.stack([continuity_loss, momentum_x_loss, momentum_y_loss], dim=1)
        # return self.data_loss(physics_loss, torch.zeros_like(physics_loss)), []
        physics_loss = [self.data_loss(continuity_loss, torch.zeros_like(continuity_loss)),
                       self.data_loss(momentum_x_loss, torch.zeros_like(momentum_x_loss)),
                       self.data_loss(momentum_y_loss, torch.zeros_like(momentum_y_loss))]
        return physics_loss, []

    def compute_adimensional_physics_loss(self, inputs: Tuple[List[torch.Tensor], List[torch.Tensor]], 
                           pred: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]):
        """
        Computes the physics-informed loss using the correct variable assignments.
        """
        # Get variables with correct assignment
        variables, missing_vars = self.assign_variables(inputs, pred)
        if missing_vars:
            return None, missing_vars
        
        # Extract and process variables
        h = variables["H*"]
        u = variables["U*"]
        v = variables["V*"]
        b = variables["B*"]
        
        Fr = variables["Fr"]
        Re = variables["Re"]
        # Ar = 40
        Ar = variables["Ar"]
        Vr = Ar
        # Vr = variables["Vr"]
        Hr = variables["Hr"]
        M = variables["M"]

        Vr = Vr.view(-1, 1, 1) 
        Fr = Fr.view(-1, 1, 1) 
        Re = Re.view(-1, 1, 1) 
        Ar = Ar.view(-1, 1, 1) 
        Hr = Hr.view(-1, 1, 1) 
        M = M.view(-1, 1, 1) 

        
       
        # Apply minimum value to H for numerical stability, and to values that can't be negative
        h = torch.clamp(h, min=self.epsilon)
        b = torch.clamp(b, min=0)
        Fr = torch.clamp(Fr, min=self.epsilon)
        Re = torch.clamp(Re, min=self.epsilon)
        Hr = torch.clamp(Hr, min=0)
        M = torch.clamp(M, min=0)
        absU = torch.sqrt(torch.clamp(u**2 + (v/Vr)**2, min=self.epsilon))


        # Compute gradients for continuity loss       
        d_hu_dx, d_hu_dy = torch.gradient(h * u, spacing=self.spacing, dim=(1, 2))
        d_hv_dx, d_hv_dy = torch.gradient(h * v, spacing=self.spacing, dim=(1, 2))
        d_h_dx, d_h_dy = torch.gradient(h, spacing=self.spacing, dim=(1, 2))
        d_u_dx, d_u_dy = torch.gradient(u, spacing=self.spacing, dim=(1, 2))
        d_v_dx, d_v_dy = torch.gradient(v, spacing=self.spacing, dim=(1, 2))
        d_z_dx, d_z_dy = torch.gradient(Hr * b + h, spacing=self.spacing, dim=(1, 2))
        d_u_dx2, d_u_dxdy = torch.gradient(d_u_dx, spacing=self.spacing, dim=(1, 2))
        d_v_dx2, d_v_dxdy = torch.gradient(d_v_dx, spacing=self.spacing, dim=(1, 2))
        d_u_dydx, d_u_dy2 = torch.gradient(d_u_dy, spacing=self.spacing, dim=(1, 2))
        d_v_dydx, d_v_dy2 = torch.gradient(d_v_dy, spacing=self.spacing, dim=(1, 2))

        continuity_loss = d_hu_dx + Ar/Vr * d_hv_dy

        advection_x = u * d_u_dx + Ar/Vr * v * d_u_dy
        advection_y = u * d_v_dx + Ar/Vr * v * d_v_dy

        pressure_x = -1/Fr**2 * d_z_dx
        pressure_y = -Ar*Vr/Fr**2 * d_z_dy

        friction_x = -M * u / (h ** (4/3)) * absU
        friction_y = -M * v / (h ** (4/3)) * absU

        diffusion_x = 1/h/Re * (d_h_dx * d_u_dx + Ar**2 * d_h_dy * d_u_dy + h * (d_u_dx2 + Ar**2 * d_u_dy2))
        diffusion_y = 1/h/Re * (d_h_dx * d_v_dx + Ar**2 * d_h_dy * d_v_dy + h * (d_v_dx2 + Ar**2 * d_v_dy2))

        momentum_x_loss =  h ** (4/3) * (advection_x - (pressure_x + friction_x + diffusion_x))
        momentum_y_loss =  h ** (4/3) * (advection_y - (pressure_y + friction_y + diffusion_y))
        # # Debug prints for continuity terms
        # print("Continuity components:")
        # print(f"d_hu_dx magnitude: mean={d_hu_dx.abs().mean()}, min={d_hu_dx.abs().min()}, max={d_hu_dx.abs().max()}")
        # print(f"d_hv_dy magnitude: mean={d_hv_dy.abs().mean()}, min={d_hv_dy.abs().min()}, max={d_hv_dy.abs().max()}")
        # if d_hu_dx.isnan().any():
        #     print("inputs")
        #     print(inputs)
        #     print("preds")
        #     print(pred)
        
        # # Debug momentum x components
        # print("\nMomentum-x components:")
        # print(f"Advection-x: mean={advection_x.abs().mean()}, min={advection_x.abs().min()}, max={advection_x.abs().max()}")
        # print(f"Pressure-x: mean={pressure_x.abs().mean()}, min={pressure_x.abs().min()}, max={pressure_x.abs().max()}")
        # print(f"Friction-x: mean={friction_x.abs().mean()}, min={friction_x.abs().min()}, max={friction_x.abs().max()}")
        # print(f"M: mean={M.abs().mean()}, min={M.abs().min()}, max={M.abs().max()}")
        # print(f"h: mean={h.abs().mean()}, min={h.abs().min()}, max={h.abs().max()}")
        # print(f"absU: mean={absU.abs().mean()}, min={absU.abs().min()}, max={absU.abs().max()}")
        # print(f"Diffusion-x: mean={diffusion_x.abs().mean()}, min={diffusion_x.abs().min()}, max={diffusion_x.abs().max()}")
        
        # # Debug momentum y components
        # print("\nMomentum-y components:")
        # print(f"Advection-y: mean={advection_y.abs().mean()}, min={advection_y.abs().min()}, max={advection_y.abs().max()}")
        # print(f"Pressure-y: mean={pressure_y.abs().mean()}, min={pressure_y.abs().min()}, max={pressure_y.abs().max()}")
        # print(f"Friction-y: mean={friction_y.abs().mean()}, min={friction_y.abs().min()}, max={friction_y.abs().max()}")
        # print(f"Diffusion-y: mean={diffusion_y.abs().mean()}, min={diffusion_y.abs().min()}, max={diffusion_y.abs().max()}")
        

        # Combine physics losses
        # physics_loss = torch.stack([continuity_loss, momentum_x_loss, momentum_y_loss], dim=1)
        # return self.data_loss(physics_loss, torch.zeros_like(physics_loss)), []
        physics_loss = [self.data_loss(continuity_loss, torch.zeros_like(continuity_loss)),
                       self.data_loss(momentum_x_loss, torch.zeros_like(momentum_x_loss)),
                       self.data_loss(momentum_y_loss, torch.zeros_like(momentum_y_loss))]
        return physics_loss, []

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
        if self.config.data.adimensional:
            required_vars = {"H*", "U*", "V*", "B*","Fr","Hr","Re","M", "Ar"} # "Ar","Vr" removed because they are constant in this case
        else:
            required_vars = {"H", "U", "V", "B", "n", "nut"}
        if not required_vars.issubset(set(self.input_vars).union(set(self.output_vars))):
            missing = required_vars - set(self.input_vars).union(set(self.output_vars))
            return None, list(missing)

    
        # Process input variables
        all_inputs = (field_inputs if field_inputs is not None else []) + (scalar_inputs if scalar_inputs is not None else [])
        sorted_input_names = [var for var in self.input_vars if var in self.config.data.non_scalars] + [var for var in self.input_vars if var in self.config.data.scalars]
        for var, tensor in zip(sorted_input_names, all_inputs):
            if self.config.data.normalize_input:
                tensor = self.dataset._denormalize(tensor, var)
            # Reshape scalar inputs to (batch_size) if they are scalar
            if var in self.config.data.scalars:
                tensor = tensor.squeeze() if tensor.dim() > 1 else tensor
            variables[var] = tensor
    
        # Separate processing for field_pred and scalar_pred
        sorted_output_non_scalar_names = [var for var in self.output_vars if var in self.config.data.non_scalars]
        if field_pred is not None:
            # Process field_pred tensor, unbinding along dim=1 (the variable dimension for field variables)
            for i, (var, tensor_slice) in enumerate(zip(sorted_output_non_scalar_names, torch.unbind(field_pred, dim=1))):
                if self.config.data.normalize_output:
                    tensor_slice = self.dataset._denormalize(tensor_slice,var)
                variables[var] = tensor_slice
                
        sorted_output_scalar_names = [var for var in self.output_vars if var in self.config.data.scalars]
        if scalar_pred is not None:
            # Process scalar_pred tensor, unbinding along dim=1 (the variable dimension for scalar variables)
            for i, (var, tensor_slice) in enumerate(zip(sorted_output_scalar_names, torch.unbind(scalar_pred, dim=1))):
                if self.config.data.normalize_output:
                    tensor_slice = self.dataset._denormalize(tensor_slice, var)
                # Ensure output scalar variables have shape (batch_size)
                tensor_slice = tensor_slice.squeeze() if tensor_slice.dim() > 1 else tensor_slice
                variables[var] = tensor_slice
    
        return variables, missing_vars