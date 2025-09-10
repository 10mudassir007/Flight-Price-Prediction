from src.predict import process_input
from src.predict import load_model,load_encoders,make_prediction
import warnings

warnings.filterwarnings('ignore')
raw_input = {
    "airline":"Not Specified",
    "source_city": "Karachi",
    "departure_time": "Morning",
    "stops": "zero",
    "arrival_time": "Afternoon",
    "destination_city": "Moscow",
    "class": "Economy",
    "duration": 2.5,
}



model = load_model()
encoders = load_encoders()
processed_df = process_input(raw_input)
predicted_price = make_prediction(model,processed_df,encoders)
print(f"Airline: {processed_df['airline'].iloc[0]}, {predicted_price} PKR")