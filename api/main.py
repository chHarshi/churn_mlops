from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

from api.schemas import CustomerFeatures, PredictionResponse
from src.churn.predict import predict_one, get_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()  # load once at startup, same reasoning as before
    yield        # app runs here
    # (nothing needed on shutdown for this app)


app = FastAPI(title="Churn Prediction API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    try:
        result = predict_one(customer.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")