import pandas as pd
import pickle
import os
from src.data import get_lat_long, haversine

def load_model(model_name='random_forest'):
    """Load a saved model from disk."""
    with open(f"models/{model_name}.pkl", "rb") as f:
        return pickle.load(f)

def load_encoders(enc_dir="models/encoders"):
    """Load all saved encoders from encoders directory."""
    encoders = {}
    if os.path.exists(enc_dir):
        for file in os.listdir(enc_dir):
            if file.endswith("_encoder.pkl"):
                col = file.replace("_encoder.pkl", "")
                with open(os.path.join(enc_dir, file), "rb") as f:
                    encoders[col] = pickle.load(f)
    return encoders

def process_input(raw_input: dict):
    df = pd.DataFrame([raw_input])
    
    df['source_latitude'], df['source_longitude'] = zip(*df['source_city'].map(get_lat_long))
    df['destination_latitude'], df['destination_longitude'] = zip(*df['destination_city'].map(get_lat_long))

    df['Distance'] = df.apply(
        lambda row: haversine(
            row['source_latitude'], row['source_longitude'],
            row['destination_latitude'], row['destination_longitude']
        ), axis=1
    )

    df = df.drop(["source_city", "destination_city"], axis=1)
    return df


def make_prediction(model, test_df, encoders=None):
    if test_df['airline'].iloc[0] == "Not Specified":
        if test_df['Distance'].iloc[0] <= 1500:
            test_df['airline'].iloc[0] = "Vistara" # 1
        elif 1500 < test_df['Distance'].iloc[0] <= 2100:
            test_df['airline'].iloc[0] = "SpiceJet" # 2
        elif 2100 < test_df['Distance'].iloc[0] <= 3000:
            test_df['airline'].iloc[0] = "Indigo" # 3
        elif 3000 <  test_df['Distance'].iloc[0] <= 3800:
            test_df['airline'].iloc[0] = "Air_India" # 3
        elif 3800 < test_df['Distance'].iloc[0] <= 4600:
            test_df['airline'].iloc[0] = "GO_FIRST" # 4
        else:
            test_df['airline'].iloc[0] = "AirAsia" # 5
    x = 1
    
    if test_df['airline'].iloc[0] == "Vistara":
        x = 1
    elif test_df['airline'].iloc[0] == "SpiceJet":
        x = 2
    elif test_df['airline'].iloc[0] == "Indigo":
        x = 3
    elif test_df['airline'].iloc[0] == "Air_India":
        x = 3.5
    elif test_df['airline'].iloc[0] == "GO_FIRST":
        x = 4
    else:
        x = 5
        
    model_features = [
    'airline','departure_time','stops','arrival_time','class',
    'duration','source_longitude','destination_longitude',
    'source_latitude','destination_latitude','Distance'
]
    
    y = 1
    if test_df['class_'].iloc[0] == 'Business':
        y = 0.25
    
    test_df = test_df[model_features]

    if encoders:
        for col, encoder in encoders.items():
            test_df[col] = encoder.transform(test_df[col])
    
    
    return model.predict(test_df)[0] * 1.131 * x * 3.26 * y
