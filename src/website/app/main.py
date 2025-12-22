# src/website/app/main.py
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .model_loader import model, postprocess_output, preprocess_input
from .schemas import ModelInput, ModelOutput

app = FastAPI(title="FNO Surrogate Model API")

# Define paths relative to this file
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# --- Mount Static Files (CSS, JS) ---
# This makes the content of the 'static' folder available at the URL '/static'
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --- Setup Template Engine for HTML ---
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# --- CORS (Cross-Origin Resource Sharing) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Endpoints ---


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serves the main index.html page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict/", response_model=ModelOutput)
async def predict(data: ModelInput):
    """Endpoint to receive data and return a model prediction."""
    try:
        # 1. Preprocess the input from the request
        model_inputs = preprocess_input(data.dict())

        # 2. Perform inference
        with torch.no_grad():
            raw_predictions = model(model_inputs)

        # 3. Postprocess the output
        results = postprocess_output(raw_predictions)
        return ModelOutput(**results)

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        import traceback

        print(f"Error during prediction: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"An internal error occurred: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Simple endpoint to check if the server is running."""
    return {"status": "ok"}
