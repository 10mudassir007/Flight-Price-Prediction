# api.py

from fastapi import APIRouter, Request
from pydantic import BaseModel
from src.predict import process_input, make_prediction
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)
router = APIRouter()

class PredictionRequest(BaseModel):
    airline: str
    source_city: str
    departure_time: str
    stops: str
    arrival_time: str
    destination_city: str
    class_: str
    duration: float

@router.post("/predict")
async def predict_flight_price(request: Request, payload: PredictionRequest):
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    raw_input = {
        "airline": payload.airline,
        "source_city": payload.source_city,
        "departure_time": payload.departure_time,
        "stops": payload.stops,
        "arrival_time": payload.arrival_time,
        "destination_city": payload.destination_city,
        "class": payload.class_,
        "duration": payload.duration
    }

    logger.info(f"[{timestamp}] Incoming request: {raw_input}")

    model = request.app.state.model
    encoders = request.app.state.encoders

    processed_df = process_input(raw_input)
    predicted_price = make_prediction(model, processed_df, encoders)
    duration = round(time.time() - start_time, 4)

    response = {
        "predicted_price_pkr": round(predicted_price, 2),
        "airline": processed_df['airline'].iloc[0]
    }

    logger.info(f"[{timestamp}] Prediction: {response} | Time taken: {duration}s")

    return response
