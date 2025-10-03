from pydantic import BaseModel


class ModelInput(BaseModel):
    # Define what your API expects. This MUST match what preprocess_input handles.
    scalar_features: list[float] | None = None
    # Example for field data:
    field_data_flat: list[float] | None = None  # Or structure for file upload
    # field_data_filename: Optional[str] = None # if handling uploads


class ModelOutput(BaseModel):
    # Define the structure of your API's response.
    scalar_predictions: list[float] | None = None
    field_predictions_shape: list[int] | None = None
    # field_predictions_flat: Optional[List[float]] = None
    field_predictions_info: str | None = None
    error: str | None = None
