from sklearn.datasets import fetch_california_housing
import pandas as pd
import os

def load_data(save_path='data/raw/california_housing.csv'):
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Data saved to {save_path}")

if __name__ == "__main__":
    load_data()