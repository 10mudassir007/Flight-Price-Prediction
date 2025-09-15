import mlflow
from pathlib import Path

# local folder for runs
MLFLOW_TRACKING_URI = Path("mlruns").absolute().as_uri()

def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("flight_price_prediction")
