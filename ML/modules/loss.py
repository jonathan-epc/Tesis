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
        use_physics_loss: bool = False,
        normalize_output: bool = True, # Add this argument
        epsilon: float = 1e-5
    ):
        super(PhysicsInformedLoss, self).__init__()
        self.g = g
        self.use_physics_loss = use_physics_loss
        self.normalize_output = normalize_output
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
        if not self.use_physics_loss:
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
        
        # Combine losses with dynamic weighting (if ReLoBRaLo is active and use_physics_loss is True)
        if self.use_physics_loss:
             total_loss = (
                 self.lambdas[0] * total_data_loss +
                 self.lambdas[1] * continuity_loss +
                 self.lambdas[2] * momentum_x_loss +
                 self.lambdas[3] * momentum_y_loss
             )
        else: # Should not happen if check above works, but good practice
             total_loss = total_data_loss
        
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
        Assigns variables following the working reference code pattern,
        respecting trial-specific normalization settings.
        """
        variables = {}
        missing_vars = []
        field_inputs, scalar_inputs = inputs
        field_pred, scalar_pred = pred
    
        # First check if we have all required variables in input_vars and output_vars
        if self.config.data.adimensional:
            # Adjusted required vars based on your comment in config.yaml
            # Assuming Ar and Vr might be constants or handled elsewhere if not in inputs/outputs
            required_vars = {"H*", "U*", "V*", "B*", "Fr", "Hr", "Re", "M", "Ar"}
        else:
            required_vars = {"H", "U", "V", "B", "n", "nut"}
    
        available_vars = set(self.input_vars).union(set(self.output_vars))
        if not required_vars.issubset(available_vars):
            missing = required_vars - available_vars
            # Log the warning here as well
            logger.warning(f"Missing required variables for physics loss calculation: {missing}")
            return None, list(missing) # Return None for variables dict to signal failure
    
    
        # Process input variables
        all_inputs = (field_inputs if field_inputs is not None else []) + (scalar_inputs if scalar_inputs is not None else [])
        # Correctly order input names based on how dataset constructs them (fields first, then scalars)
        sorted_input_names = [var for var in self.input_vars if var in self.config.data.non_scalars] + \
                             [var for var in self.input_vars if var in self.config.data.scalars]
    
        # Check if the number of tensors matches the number of expected input variables
        if len(all_inputs) != len(sorted_input_names):
             logger.error(f"Mismatch between number of input tensors ({len(all_inputs)}) and input variable names ({len(sorted_input_names)}). Check data loading and config.")
             # Handle error appropriately, maybe raise ValueError or return missing
             return None, list(required_vars) # Indicate failure
    
        for var, tensor in zip(sorted_input_names, all_inputs):
            # Denormalize INPUTS based on the trial's setting (self.normalize_input)
            # The dataset provides data normalized based on config.data.normalize_input.
            # We denormalize here *only if* the dataset provided normalized data.
            if self.config.data.normalize_input: # Check if dataset provided normalized data
                 tensor_denorm = self.dataset._denormalize(tensor, var)
            else:
                 tensor_denorm = tensor # Already in original scale
    
            # If the *trial* expects *normalized* inputs, use the original tensor.
            # If the *trial* expects *unnormalized* inputs, use the denormalized tensor.
            # The self.normalize_input flag tells us what the *trial* expects.
            # This logic seems counter-intuitive. Let's rethink:
            # The physics equations expect *real-world* values.
            # So, we *always* need to denormalize if the dataset provided normalized inputs.
            # The 'normalize_input' hyperparameter tune doesn't make sense here,
            # physics requires real values. Let's assume we always denormalize inputs
            # if they came normalized from the dataset.
            if self.config.data.normalize_input:
                tensor_to_use = self.dataset._denormalize(tensor.clone(), var) # Use .clone()
            else:
                tensor_to_use = tensor.clone()
    
            # Reshape scalar inputs AFTER potential denormalization
            if var in self.config.data.scalars:
                # Ensure it's at least 1D for viewing later
                tensor_to_use = tensor_to_use.squeeze() if tensor_to_use.dim() > 1 else tensor_to_use
                if tensor_to_use.dim() == 0: # If squeeze resulted in 0D tensor
                    tensor_to_use = tensor_to_use.unsqueeze(0) # Make it 1D (batch size 1 case)
    
    
            variables[var] = tensor_to_use
    
        # Separate processing for field_pred and scalar_pred
        # Denormalize OUTPUTS based on the trial's setting (self.normalize_output)
        sorted_output_non_scalar_names = [var for var in self.output_vars if var in self.config.data.non_scalars]
        if field_pred is not None:
            if field_pred.shape[1] != len(sorted_output_non_scalar_names):
                 logger.error(f"Mismatch between field_pred channels ({field_pred.shape[1]}) and non-scalar output variables ({len(sorted_output_non_scalar_names)}).")
                 return None, list(required_vars) # Indicate failure
    
            for i, (var, tensor_slice) in enumerate(zip(sorted_output_non_scalar_names, torch.unbind(field_pred, dim=1))):
                # If the trial used normalized outputs (self.normalize_output is True),
                # then the prediction (tensor_slice) is normalized and needs denormalizing for physics.
                if self.normalize_output:
                    tensor_slice_denorm = self.dataset._denormalize(tensor_slice.clone(), var) # Use .clone()
                    variables[var] = tensor_slice_denorm
                else:
                    # If the trial used unnormalized outputs, the prediction is already in real scale.
                    variables[var] = tensor_slice.clone()
    
        sorted_output_scalar_names = [var for var in self.output_vars if var in self.config.data.scalars]
        if scalar_pred is not None:
            if scalar_pred.shape[1] != len(sorted_output_scalar_names):
                 logger.error(f"Mismatch between scalar_pred channels ({scalar_pred.shape[1]}) and scalar output variables ({len(sorted_output_scalar_names)}).")
                 return None, list(required_vars) # Indicate failure
    
            for i, (var, tensor_slice) in enumerate(zip(sorted_output_scalar_names, torch.unbind(scalar_pred, dim=1))):
                # Same logic as for field predictions
                if self.normalize_output:
                    tensor_slice_denorm = self.dataset._denormalize(tensor_slice.clone(), var) # Use .clone()
                else:
                    tensor_slice_denorm = tensor_slice.clone()
    
                # Reshape scalar outputs AFTER potential denormalization
                # Ensure it's at least 1D for viewing later
                tensor_slice_denorm = tensor_slice_denorm.squeeze() if tensor_slice_denorm.dim() > 1 else tensor_slice_denorm
                if tensor_slice_denorm.dim() == 0:
                     tensor_slice_denorm = tensor_slice_denorm.unsqueeze(0) # Make it 1D
    
                variables[var] = tensor_slice_denorm
    
        # Final check if all required variables are present in the 'variables' dict
        if not required_vars.issubset(variables.keys()):
            missing_final = required_vars - set(variables.keys())
            logger.error(f"Failed to assign all required variables for physics loss. Missing: {missing_final}")
            return None, list(missing_final)
    
        return variables, [] # Return empty list for missing_vars on success