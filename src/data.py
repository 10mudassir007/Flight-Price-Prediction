import os
import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split
from .utils import get_lat_long,haversine
import pickle
from sklearn.preprocessing import LabelEncoder

def load_dataset():
    """Download and load dataset from Kaggle."""
    path = kagglehub.dataset_download("shubhambathwal/flight-price-prediction")
    
    ds_path = [os.path.join(path, p) for p in os.listdir(path)]
    df = pd.read_csv(ds_path[1])
    
    if "raw_data.csv" not in os.listdir("data"):
        df.to_csv("data/raw_data.csv",index=False)
    return df

def add_coordinates(df):
    """Add latitude/longitude columns for source and destination cities."""
    cities = pd.concat([df['source_city'], df['destination_city']]).unique()
    coordinates = {city: get_lat_long(city + ",India") for city in cities}
    
    df['source_longitude'] = df['source_city'].map(lambda x: coordinates.get(x, (0,0))[1])
    df['destination_longitude'] = df['destination_city'].map(lambda x: coordinates.get(x, (0,0))[1])
    df['source_latitude'] = df['source_city'].map(lambda x: coordinates.get(x, (0,0))[0])
    df['destination_latitude'] = df['destination_city'].map(lambda x: coordinates.get(x, (0,0))[0])
    
    df['Distance'] = df.apply(
        lambda row: haversine(
            row['source_latitude'], row['source_longitude'],
            row['destination_latitude'], row['destination_longitude']
        ), axis=1
    )
    return df

def prepare_features(df):
    """Keep only selected features for training."""
    cols_to_keep = [
        'airline','departure_time','stops','arrival_time','class','duration','price',
        'source_longitude','destination_longitude','source_latitude','destination_latitude','Distance'
    ]
    df = df[cols_to_keep]
    
    cols_to_encode = ['airline','departure_time','stops','arrival_time','class']
    return df, cols_to_encode



def encode_data(train_df, test_df, cols_to_encode, encoders_dir="models/encoders"):
    """Label encode categorical features and save encoders individually as .pkl files."""
    os.makedirs(encoders_dir, exist_ok=True)

    for col in cols_to_encode:
        encoder = LabelEncoder()
        train_df[col] = encoder.fit_transform(train_df[col])
        test_df[col] = encoder.transform(test_df[col])

        # Save each encoder separately
        enc_path = os.path.join(encoders_dir, f"{col}_encoder.pkl")
        with open(enc_path, "wb") as f:
            pickle.dump(encoder, f)

    return train_df, test_df


def train_test_split_data(df, test_size=0.1, random_state=42):
    """Split dataset into train and test."""
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df, test_df

def save_processed_data(df, filename="data/processed_data.csv"):
    """Save processed dataset to file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"Processed data saved to {filename}")

def load_processed_dataset():
    try:
        df = pd.read_csv(r"E:\Mudassir\KONG\personal\Flight-Price-Prediction\data\processed_data.csv")
        return df
    except:
        return "Data is not processed yet"
    