import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataReader: 
    def __init__(self, csv_path, window_size=5):
        self.csv_path = csv_path
        self.window_size = window_size
        self.scaler = MinMaxScaler()

    def read(self) -> np.ndarray:
        df = pd.read_csv(self.csv_path)
        df = df[["cpu_percent", "mem_mb"]]
        scaled = self.scaler.fit_transform(df)
        return scaled

    def create_sequences(self) -> tuple[np.ndarray, np.ndarray]:
        data = self.read()
        X, y = [], []
        for i in range(len(data) - self.window_size):
            X.append(data[i:i+self.window_size])
            y.append(data[i+self.window_size])
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        return X, y

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)