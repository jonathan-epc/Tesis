# src/ML/modules/metrics.py


import pandas as pd
import torch
import torch.nn.functional as F


def calculate_metrics_on_tensors(
    pred: torch.Tensor, target: torch.Tensor
) -> dict[str, float]:
    """Calculates a dictionary of regression metrics for flattened tensors.

    Args:
        pred (torch.Tensor): The prediction tensor.
        target (torch.Tensor): The ground truth tensor.

    Returns:
        Dict[str, float]: A dictionary of calculated metrics (MSE, RMSE, MAE, R², SMAPE).
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    # Use a small epsilon to prevent division by zero in R² and SMAPE
    epsilon = 1e-9

    mse = F.mse_loss(pred_flat, target_flat).item()
    mae = F.l1_loss(pred_flat, target_flat).item()

    ss_tot = torch.sum((target_flat - torch.mean(target_flat)) ** 2)
    ss_res = torch.sum((target_flat - pred_flat) ** 2)
    r2 = (1 - ss_res / (ss_tot + epsilon)).item()

    smape_demoninator = torch.abs(target_flat) + torch.abs(pred_flat) + epsilon
    smape = (
        torch.mean(2 * torch.abs(pred_flat - target_flat) / smape_demoninator) * 100
    ).item()

    return {
        "mse": mse,
        "rmse": mse**0.5,
        "mae": mae,
        "r2": r2,
        "smape": smape,
    }


def evaluate_predictions(
    predictions: tuple[torch.Tensor | None, torch.Tensor | None],
    targets: tuple[torch.Tensor | None, torch.Tensor | None],
    output_fields: list[str],
    output_scalars: list[str],
) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    """Calculates metrics for a full set of predictions and targets.

    Args:
        predictions (Tuple): A tuple of (field_predictions, scalar_predictions).
        targets (Tuple): A tuple of (field_targets, scalar_targets).
        output_fields (List[str]): List of names for the field variables.
        output_scalars (List[str]): List of names for the scalar variables.

    Returns:
        A tuple containing:
        - A dictionary of aggregated metrics for each variable and overall.
        - A pandas DataFrame with detailed per-case metrics.
    """
    field_preds, scalar_preds = predictions
    field_targs, scalar_targs = targets

    all_metrics: dict[str, dict[str, float]] = {}
    per_case_data = []

    # Process field variables
    if field_preds is not None and field_targs is not None:
        for i, name in enumerate(output_fields):
            pred_var, targ_var = field_preds[:, i], field_targs[:, i]
            all_metrics[name] = calculate_metrics_on_tensors(pred_var, targ_var)
            for j in range(pred_var.shape[0]):  # Iterate through batch
                case_metrics = calculate_metrics_on_tensors(pred_var[j], targ_var[j])
                per_case_data.append(
                    {"case_id": j, "variable": name, "type": "field", **case_metrics}
                )

    # Process scalar variables
    if scalar_preds is not None and scalar_targs is not None:
        for i, name in enumerate(output_scalars):
            pred_var, targ_var = scalar_preds[:, i], scalar_targs[:, i]
            all_metrics[name] = calculate_metrics_on_tensors(pred_var, targ_var)
            for j in range(pred_var.shape[0]):
                case_metrics = calculate_metrics_on_tensors(pred_var[j], targ_var[j])
                per_case_data.append(
                    {"case_id": j, "variable": name, "type": "scalar", **case_metrics}
                )

    per_case_df = pd.DataFrame(per_case_data)
    return all_metrics, per_case_df
