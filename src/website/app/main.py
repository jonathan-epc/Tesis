import torch
from fastapi import FastAPI
from fastapi.middleware.cors import (
    CORSMiddleware,
)

# For allowing requests from different origins (e.g. local frontend)
from .model_loader import (
    DEVICE,
    model,
    postprocess_output,
    preprocess_input,
)

# Import from model_loader.py
from .schemas import ModelInput, ModelOutput  # Import Pydantic models

app = FastAPI(title="My Neural Network API")

# --- CORS (Cross-Origin Resource Sharing) ---
# Allow requests from your frontend if it's served on a different port/domain during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],  # Or specify origins: ["http://localhost:3000", "http://127.0.0.1:5500"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Mount static files and templates (Optional, if serving HTML from FastAPI) ---
# app.mount("/static", StaticFiles(directory="static"), name="static")
# templates = Jinja2Templates(directory="templates")

# @app.get("/")
# async def read_root(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict/", response_model=ModelOutput)
async def predict(data: ModelInput):
    try:
        print(f"Received data: {data.dict()}")
        # 1. Preprocess the input from the request
        # The 'data' object is already parsed by Pydantic from JSON
        model_inputs = preprocess_input(data.dict())

        # 2. Perform inference
        with torch.no_grad():
            raw_predictions = model(*model_inputs)  # model expects tuple of lists

        # 3. Postprocess the output
        results = postprocess_output(raw_predictions)
        return ModelOutput(**results)

    except Exception as e:
        import traceback

        print(f"Error during prediction: {e}")
        traceback.print_exc()
        # Return a Pydantic model even for errors
        return ModelOutput(error=str(e))
        # Or raise HTTPException:
        # raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "ok", "device": str(DEVICE)}


if __name__ == "__main__":
    import uvicorn

    # This is for running directly, for production use Gunicorn with Uvicorn workers
    uvicorn.run(app, host="0.0.0.0", port=8000)
