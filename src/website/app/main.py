# src/website/app/main.py
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# We will create a model handler to manage all 12 models
from .model_loader import ModelHandler
from .schemas import ModelInput, ModelOutput

app = FastAPI(title="FNO Surrogate Model API")
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# --- Initialize Model Handler ---
# This will find and load all models in the models_store on startup
model_handler = ModelHandler(model_store_path=BASE_DIR / "models_store")

# --- Mount Static Files & Templates ---
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/models")
async def get_models():
    """Returns a list of available models and their configs."""
    return model_handler.get_available_models()


@app.post("/predict/{model_key}", response_model=ModelOutput)
async def predict(model_key: str, data: ModelInput):
    """Endpoint to run prediction for a specific model."""
    try:
        results = model_handler.predict(model_key, data.dict())
        return ModelOutput(**results)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"Model key '{model_key}' not found."
        ) from None
    except Exception as e:
        import traceback

        print(f"Error during prediction: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"An internal error occurred: {str(e)}"
        ) from e


@app.get("/health")
async def health_check():
    return {"status": "ok", "models_loaded": len(model_handler.get_available_models())}
