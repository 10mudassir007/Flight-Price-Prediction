import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager
from src.predict import load_model, load_encoders
from api import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🔧 Loading model and encoders...")
    app.state.model = load_model()
    app.state.encoders = load_encoders()
    logger.info("✅ Model and encoders loaded successfully.")
    yield
    logger.info("🧹 Shutting down app and clearing model/encoders.")
    app.state.model = None
    app.state.encoders = None

app = FastAPI(
    title="Flight Price Prediction API",
    version="1.0",
    lifespan=lifespan
)

app.include_router(router)
