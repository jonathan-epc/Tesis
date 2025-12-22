# src/website/app/schemas.py
from typing import Dict, List, Optional

from pydantic import BaseModel


class ModelInput(BaseModel):
    """Defines the structure of the data the API expects to receive."""

    # The direct problem model expects 4 scalar features: H0, Q0, n, nut
    scalar_features: List[float]
    # The direct problem model expects 1 field (Bed Geometry 'B')
    # with 11x401 = 4411 points, flattened into a single list.
    field_data_flat: List[float]


class ModelOutput(BaseModel):
    """Defines the structure of the data the API will send back."""

    # The direct problem model predicts 3 scalar outputs (empty in this case)
    scalar_predictions: Optional[List[float]] = None
    # And 3 field outputs (H, U, V)
    field_predictions_info: Optional[str] = None
    field_predictions_shape: Optional[List[int]] = None
    # This will hold the actual data for plotting, e.g., {"H": [...], "U": [...]}
    field_predictions_flat: Optional[Dict[str, List[float]]] = None
    # We will also include a field for any errors that occur.
    error: Optional[str] = None
