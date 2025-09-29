from typing import Any

import torch
import torch.nn as nn
from loguru import logger


class PhysicsInformedLoss(nn.Module):
    """Physics-informed neural network loss function with ReLoBRaLo weighting.

    This loss function combines data-driven loss with physics-based loss terms
    derived from fluid dynamics equations (continuity and momentum). It implements
    the ReLoBRaLo (Relative Loss Balancing with Random Lookback) algorithm for
    dynamic loss weighting.

    Attributes:
        g: Gravitational acceleration constant.
        use_physics_loss: Whether to include physics-based loss terms.
        normalize_output: Whether outputs are normalized and need denormalization.
        data_loss: Huber loss function for data terms.
        input_vars: List of input variable names.
        output_vars: List of output variable names.
        config: Configuration object containing model and data parameters.
        dataset: Dataset object for denormalization operations.
        epsilon: Small constant for numerical stability.
        alpha: Exponential moving average parameter for ReLoBRaLo.
        temperature: Temperature parameter for softmax in ReLoBRaLo.
        rho: Probability parameter for random lookback in ReLoBRaLo.
        call_count: Number of times forward pass has been called.
        spacing: Grid spacing for gradient calculations (dy, dx).
        lambdas: Current loss weights for ReLoBRaLo.
        last_losses: Loss values from the previous iteration.
        init_losses: Initial loss values for random lookback.
    """

    def __init__(
        self,
        input_vars: list[str],
        output_vars: list[str],
        config: Any,
        dataset: Any,
        g: float = 9.81,
        alpha: float = 0.999,
        temperature: float = 0.1,
        rho: float = 0.99,
        use_physics_loss: bool = False,
        normalize_output: bool = True,
        epsilon: float = 1e-5,
    ) -> None:
        """Initialize the PhysicsInformedLoss module.

        Args:
            input_vars: List of input variable names.
            output_vars: List of output variable names.
            config: Configuration object containing model and data parameters.
            dataset: Dataset object for denormalization operations.
            g: Gravitational acceleration constant. Defaults to 9.81.
            alpha: Exponential moving average parameter for ReLoBRaLo. Defaults to 0.999.
            temperature: Temperature parameter for softmax in ReLoBRaLo. Defaults to 0.1.
            rho: Probability parameter for random lookback in ReLoBRaLo. Defaults to 0.99.
            use_physics_loss: Whether to include physics-based loss terms. Defaults to False.
            normalize_output: Whether outputs are normalized. Defaults to True.
            epsilon: Small constant for numerical stability. Defaults to 1e-5.
        """
        super().__init__()
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

        # Calculate grid spacing
        dx = self.config.channel.length / (self.config.mesh.num_points_x - 1)
        dy = self.config.channel.width / (self.config.mesh.num_points_y - 1)

        if self.config.data.is_adimensional:
            dx_norm = dx / self.config.channel.length
            dy_norm = dy / self.config.channel.width
            self.spacing = (dy_norm, dx_norm)
        else:
            self.spacing = (dy, dx)

        logger.debug(f"Physics loss initialized with spacing: {self.spacing}")

        # Initialize ReLoBRaLo variables
        self.lambdas = torch.ones(
            4
        )  # Four loss terms: data, continuity, momentum_x, momentum_y
        self.last_losses = torch.ones(4)
        self.init_losses = torch.ones(4)

    def forward(
        self,
        inputs: tuple[list[torch.Tensor], list[torch.Tensor]],
        pred: tuple[torch.Tensor | None, torch.Tensor | None],
        target: tuple[list[torch.Tensor], list[torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the combined physics-informed loss.

        Args:
            inputs: Tuple of (field_inputs, scalar_inputs) containing input tensors.
            pred: Tuple of (field_predictions, scalar_predictions) from the model.
            target: Tuple of (field_targets, scalar_targets) containing ground truth.

        Returns:
            Tuple containing:
                - total_loss: Combined weighted loss
                - data_loss: Data-driven loss component
                - continuity_loss: Continuity equation loss
                - momentum_x_loss: X-momentum equation loss
                - momentum_y_loss: Y-momentum equation loss
        """
        field_target, scalar_target = target
        field_pred, scalar_pred = pred

        # Compute data loss
        total_data_loss = 0.0
        if field_pred is not None and field_target:
            field_target_tensor = torch.stack(field_target, dim=1)
            if field_pred.shape != field_target_tensor.shape:
                raise ValueError(
                    f"Shape mismatch: field_pred {field_pred.shape}, field_target {field_target_tensor.shape}"
                )
            total_data_loss += self.data_loss(field_pred, field_target_tensor)

        if scalar_pred is not None and scalar_target:
            scalar_target_tensor = torch.stack(scalar_target, dim=1)
            if scalar_pred.shape != scalar_target_tensor.shape:
                raise ValueError(
                    f"Shape mismatch: scalar_pred {scalar_pred.shape}, scalar_target {scalar_target_tensor.shape}"
                )
            total_data_loss += self.data_loss(scalar_pred, scalar_target_tensor)

        # Check if physics loss should be computed
        zero_tensor = torch.tensor(0.0, device=total_data_loss.device)
        if not self.use_physics_loss:
            return (
                total_data_loss,
                total_data_loss,
                zero_tensor,
                zero_tensor,
                zero_tensor,
            )

        # Compute physics loss
        compute_physics = (
            self.compute_adimensional_physics_loss
            if self.config.data.is_adimensional
            else self.compute_physics_loss
        )
        physics_loss, missing_vars = compute_physics(inputs, pred)

        if missing_vars:
            logger.warning(
                f"Missing variables for physics loss: {', '.join(missing_vars)}"
            )
            return (
                total_data_loss,
                total_data_loss,
                zero_tensor,
                zero_tensor,
                zero_tensor,
            )

        continuity_loss, momentum_x_loss, momentum_y_loss = physics_loss

        # Update ReLoBRaLo lambdas
        self.update_relobralo_lambdas(
            total_data_loss, continuity_loss, momentum_x_loss, momentum_y_loss
        )

        # Combine losses with dynamic weighting
        if self.use_physics_loss:
            total_loss = (
                self.lambdas[0] * total_data_loss
                + self.lambdas[1] * continuity_loss
                + self.lambdas[2] * momentum_x_loss
                + self.lambdas[3] * momentum_y_loss
            )
        else:
            total_loss = total_data_loss

        return (
            total_loss,
            total_data_loss,
            continuity_loss,
            momentum_x_loss,
            momentum_y_loss,
        )

    def update_relobralo_lambdas(
        self,
        total_data_loss: torch.Tensor,
        continuity_loss: torch.Tensor,
        momentum_x_loss: torch.Tensor,
        momentum_y_loss: torch.Tensor,
    ) -> None:
        """Update ReLoBRaLo loss weights based on relative loss magnitudes.

        Implements the ReLoBRaLo algorithm that dynamically adjusts loss weights
        based on the relative magnitudes of different loss components, using
        exponential moving averages and random lookback for stability.

        Args:
            total_data_loss: Current data loss value.
            continuity_loss: Current continuity equation loss value.
            momentum_x_loss: Current x-momentum equation loss value.
            momentum_y_loss: Current y-momentum equation loss value.
        """
        losses = torch.tensor(
            [
                total_data_loss.item(),
                continuity_loss.item(),
                momentum_x_loss.item(),
                momentum_y_loss.item(),
            ]
        )

        # Compute alpha and rho based on iteration
        alpha = (
            1.0
            if self.call_count == 0
            else (0.0 if self.call_count == 1 else self.alpha)
        )
        rho = (
            1.0
            if self.call_count <= 1
            else torch.bernoulli(torch.tensor(self.rho)).item()
        )

        # Compute lambdas_hat (current iteration)
        lambdas_hat = losses / (self.last_losses * self.temperature + self.epsilon)
        lambdas_hat = torch.softmax(lambdas_hat - torch.max(lambdas_hat), dim=0) * len(
            losses
        )

        # Compute init_lambdas_hat (random lookback)
        init_lambdas_hat = losses / (self.init_losses * self.temperature + self.epsilon)
        init_lambdas_hat = torch.softmax(
            init_lambdas_hat - torch.max(init_lambdas_hat), dim=0
        ) * len(losses)

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

    def compute_physics_loss(
        self,
        inputs: tuple[list[torch.Tensor], list[torch.Tensor]],
        pred: tuple[torch.Tensor | None, torch.Tensor | None],
    ) -> tuple[list[torch.Tensor] | None, list[str]]:
        """Compute physics-based loss terms using dimensional shallow water equations.

        Computes loss terms based on the continuity equation and momentum equations
        for shallow water flow, using dimensional variables.

        Args:
            inputs: Tuple of (field_inputs, scalar_inputs) containing input tensors.
            pred: Tuple of (field_predictions, scalar_predictions) from the model.

        Returns:
            Tuple containing:
                - physics_loss: List of [continuity_loss, momentum_x_loss, momentum_y_loss]
                  or None if missing variables.
                - missing_vars: List of variable names that are missing for computation.
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

        # Apply minimum values for numerical stability
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
        d_z_dx, d_z_dy = torch.gradient(h + b, spacing=self.spacing, dim=(1, 2))

        # Second-order gradients for diffusion terms
        d_u_dx2, d_u_dxdy = torch.gradient(d_u_dx, spacing=self.spacing, dim=(1, 2))
        d_v_dx2, d_v_dxdy = torch.gradient(d_v_dx, spacing=self.spacing, dim=(1, 2))
        d_u_dydx, d_u_dy2 = torch.gradient(d_u_dy, spacing=self.spacing, dim=(1, 2))
        d_v_dydx, d_v_dy2 = torch.gradient(d_v_dy, spacing=self.spacing, dim=(1, 2))

        # Continuity equation: ∂(hu)/∂x + ∂(hv)/∂y = 0
        continuity_loss = d_hu_dx + d_hv_dy

        # Momentum equation components
        advection_x = u * d_u_dx + v * d_u_dy
        advection_y = u * d_v_dx + v * d_v_dy

        pressure_x = -self.g * d_z_dx
        pressure_y = -self.g * d_z_dy

        friction_x = -self.g * (n**2) / (h ** (4 / 3)) * absU * u
        friction_y = -self.g * (n**2) / (h ** (4 / 3)) * absU * v

        diffusion_x = (
            nut / h * (d_h_dx * d_u_dx + d_h_dy * d_u_dy + h * (d_u_dx2 + d_u_dy2))
        )
        diffusion_y = (
            nut / h * (d_h_dx * d_v_dx + d_h_dy * d_v_dy + h * (d_v_dx2 + d_v_dy2))
        )

        # Momentum equations: ∂u/∂t + u∂u/∂x + v∂u/∂y = -g∂z/∂x + friction + diffusion
        momentum_x_loss = h ** (4 / 3) * (
            advection_x - (pressure_x + friction_x + diffusion_x)
        )
        momentum_y_loss = h ** (4 / 3) * (
            advection_y - (pressure_y + friction_y + diffusion_y)
        )

        physics_loss = [
            self.data_loss(continuity_loss, torch.zeros_like(continuity_loss)),
            self.data_loss(momentum_x_loss, torch.zeros_like(momentum_x_loss)),
            self.data_loss(momentum_y_loss, torch.zeros_like(momentum_y_loss)),
        ]

        return physics_loss, []

    def compute_adimensional_physics_loss(
        self,
        inputs: tuple[list[torch.Tensor], list[torch.Tensor]],
        pred: tuple[torch.Tensor | None, torch.Tensor | None],
    ) -> tuple[list[torch.Tensor] | None, list[str]]:
        """Compute physics-based loss terms using non-dimensional shallow water equations.

        Computes loss terms based on the continuity equation and momentum equations
        for shallow water flow, using non-dimensional variables with scaling parameters.

        Args:
            inputs: Tuple of (field_inputs, scalar_inputs) containing input tensors.
            pred: Tuple of (field_predictions, scalar_predictions) from the model.

        Returns:
            Tuple containing:
                - physics_loss: List of [continuity_loss, momentum_x_loss, momentum_y_loss]
                  or None if missing variables.
                - missing_vars: List of variable names that are missing for computation.
        """
        # Get variables with correct assignment
        variables, missing_vars = self.assign_variables(inputs, pred)
        if missing_vars:
            return None, missing_vars

        # Extract and process non-dimensional variables
        h = variables["H*"]
        u = variables["U*"]
        v = variables["V*"]
        b = variables["B*"]
        Fr = variables["Fr"]
        Re = variables["Re"]
        Ar = variables["Ar"]
        Vr = Ar  # Velocity ratio
        Hr = variables["Hr"]
        M = variables["M"]

        # Reshape scalar variables for broadcasting
        Vr = Vr.view(-1, 1, 1)
        Fr = Fr.view(-1, 1, 1)
        Re = Re.view(-1, 1, 1)
        Ar = Ar.view(-1, 1, 1)
        Hr = Hr.view(-1, 1, 1)
        M = M.view(-1, 1, 1)

        # Apply minimum values for numerical stability
        h = torch.clamp(h, min=self.epsilon)
        b = torch.clamp(b, min=0)
        Fr = torch.clamp(Fr, min=self.epsilon)
        Re = torch.clamp(Re, min=self.epsilon)
        Hr = torch.clamp(Hr, min=0)
        M = torch.clamp(M, min=0)

        absU = torch.sqrt(torch.clamp(u**2 + (v / Vr) ** 2, min=self.epsilon))

        # Compute gradients for non-dimensional equations
        d_hu_dx, d_hu_dy = torch.gradient(h * u, spacing=self.spacing, dim=(1, 2))
        d_hv_dx, d_hv_dy = torch.gradient(h * v, spacing=self.spacing, dim=(1, 2))
        d_h_dx, d_h_dy = torch.gradient(h, spacing=self.spacing, dim=(1, 2))
        d_u_dx, d_u_dy = torch.gradient(u, spacing=self.spacing, dim=(1, 2))
        d_v_dx, d_v_dy = torch.gradient(v, spacing=self.spacing, dim=(1, 2))
        d_z_dx, d_z_dy = torch.gradient(Hr * b + h, spacing=self.spacing, dim=(1, 2))

        # Second-order gradients for diffusion terms
        d_u_dx2, d_u_dxdy = torch.gradient(d_u_dx, spacing=self.spacing, dim=(1, 2))
        d_v_dx2, d_v_dxdy = torch.gradient(d_v_dx, spacing=self.spacing, dim=(1, 2))
        d_u_dydx, d_u_dy2 = torch.gradient(d_u_dy, spacing=self.spacing, dim=(1, 2))
        d_v_dydx, d_v_dy2 = torch.gradient(d_v_dy, spacing=self.spacing, dim=(1, 2))

        # Non-dimensional continuity equation
        continuity_loss = d_hu_dx + Ar / Vr * d_hv_dy

        # Non-dimensional momentum equation components
        advection_x = u * d_u_dx + Ar / Vr * v * d_u_dy
        advection_y = u * d_v_dx + Ar / Vr * v * d_v_dy

        pressure_x = -1 / Fr**2 * d_z_dx
        pressure_y = -Ar * Vr / Fr**2 * d_z_dy

        friction_x = -M * u / (h ** (4 / 3)) * absU
        friction_y = -M * v / (h ** (4 / 3)) * absU

        diffusion_x = (
            1
            / h
            / Re
            * (
                d_h_dx * d_u_dx
                + Ar**2 * d_h_dy * d_u_dy
                + h * (d_u_dx2 + Ar**2 * d_u_dy2)
            )
        )
        diffusion_y = (
            1
            / h
            / Re
            * (
                d_h_dx * d_v_dx
                + Ar**2 * d_h_dy * d_v_dy
                + h * (d_v_dx2 + Ar**2 * d_v_dy2)
            )
        )

        # Non-dimensional momentum equations
        momentum_x_loss = h ** (4 / 3) * (
            advection_x - (pressure_x + friction_x + diffusion_x)
        )
        momentum_y_loss = h ** (4 / 3) * (
            advection_y - (pressure_y + friction_y + diffusion_y)
        )

        physics_loss = [
            self.data_loss(continuity_loss, torch.zeros_like(continuity_loss)),
            self.data_loss(momentum_x_loss, torch.zeros_like(momentum_x_loss)),
            self.data_loss(momentum_y_loss, torch.zeros_like(momentum_y_loss)),
        ]

        return physics_loss, []

    def assign_variables(
        self,
        inputs: tuple[list[torch.Tensor], list[torch.Tensor]],
        pred: tuple[torch.Tensor | None, torch.Tensor | None],
    ) -> tuple[dict[str, torch.Tensor] | None, list[str]]:
        """Assign variables from model inputs and predictions for physics calculations.

        Maps model inputs and predictions to named variables required for physics
        computations. Handles denormalization if needed and ensures all tensors
        are in their physical units.

        Args:
            inputs: Tuple of (field_inputs, scalar_inputs) containing input tensors.
            pred: Tuple of (field_predictions, scalar_predictions) from the model.

        Returns:
            Tuple containing:
                - variables: Dictionary mapping variable names to tensors, or None if
                  missing required variables.
                - missing_vars: List of variable names that are missing for computation.
        """
        variables = {}
        field_inputs, scalar_inputs = inputs
        field_pred, scalar_pred = pred

        required_vars = (
            {"H*", "U*", "V*", "B*", "Fr", "Hr", "Re", "M", "Ar"}
            if self.config.data.is_adimensional
            else {"H", "U", "V", "B", "n", "nut"}
        )

        available_vars = set(self.input_vars).union(set(self.output_vars))
        if not required_vars.issubset(available_vars):
            missing = required_vars - available_vars
            logger.warning(
                f"Missing required variables for physics loss calculation: {missing}"
            )
            return None, list(missing)

        # Process input variables
        all_inputs = (field_inputs if field_inputs is not None else []) + (
            scalar_inputs if scalar_inputs is not None else []
        )
        sorted_input_names = (
            self.config.data.input_fields + self.config.data.input_scalars
        )

        if len(all_inputs) != len(sorted_input_names):
            logger.error(
                f"Mismatch in number of input tensors ({len(all_inputs)}) and names ({len(sorted_input_names)})."
            )
            return None, list(required_vars)

        for var, tensor in zip(sorted_input_names, all_inputs, strict=False):
            tensor_to_use = (
                self.dataset._denormalize(tensor.clone(), var)
                if self.config.data.normalize_input
                else tensor.clone()
            )
            if var in self.config.data.all_scalar_vars:
                tensor_to_use = (
                    tensor_to_use.squeeze()
                    if tensor_to_use.dim() > 1
                    else tensor_to_use
                )
                if tensor_to_use.dim() == 0:
                    tensor_to_use = tensor_to_use.unsqueeze(0)
            variables[var] = tensor_to_use

        # Process predicted variables
        sorted_output_non_scalar_names = self.config.data.output_fields
        if field_pred is not None:
            if field_pred.shape[1] != len(sorted_output_non_scalar_names):
                logger.error(
                    f"Mismatch in field_pred channels ({field_pred.shape[1]}) and non-scalar outputs ({len(sorted_output_non_scalar_names)})."
                )
                return None, list(required_vars)

            for var, tensor_slice in zip(
                sorted_output_non_scalar_names,
                torch.unbind(field_pred, dim=1),
                strict=False,
            ):
                variables[var] = (
                    self.dataset._denormalize(tensor_slice.clone(), var)
                    if self.normalize_output
                    else tensor_slice.clone()
                )

        sorted_output_scalar_names = self.config.data.output_scalars
        if scalar_pred is not None:
            if scalar_pred.shape[1] != len(sorted_output_scalar_names):
                logger.error(
                    f"Mismatch in scalar_pred channels ({scalar_pred.shape[1]}) and scalar outputs ({len(sorted_output_scalar_names)})."
                )
                return None, list(required_vars)

            for var, tensor_slice in zip(
                sorted_output_scalar_names,
                torch.unbind(scalar_pred, dim=1),
                strict=False,
            ):
                tensor_slice_denorm = (
                    self.dataset._denormalize(tensor_slice.clone(), var)
                    if self.normalize_output
                    else tensor_slice.clone()
                )
                tensor_slice_denorm = (
                    tensor_slice_denorm.squeeze()
                    if tensor_slice_denorm.dim() > 1
                    else tensor_slice_denorm
                )
                if tensor_slice_denorm.dim() == 0:
                    tensor_slice_denorm = tensor_slice_denorm.unsqueeze(0)
                variables[var] = tensor_slice_denorm

        if not required_vars.issubset(variables.keys()):
            missing_final = required_vars - set(variables.keys())
            logger.error(
                f"Failed to assign all required variables. Missing: {missing_final}"
            )
            return None, list(missing_final)

        return variables, []
