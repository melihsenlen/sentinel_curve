import torch
import torch.nn as nn
import yaml
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from architecture.data import DataReader
from architecture.model import RegressionModel

class Trainer:
    def __init__(self, config_path: str):
        self.config = self._load(config_path)
        self.reader = DataReader(self.config["data"]["csv_path"], self.config["data"]["window_size"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RegressionModel(input_size=2).to(self.device)

    @staticmethod
    def _load(path: str) -> dict:
        with open(path) as y:
            return yaml.safe_load(y)

    def _build(self) -> DataLoader:
        X, y = self.reader.create_sequences()
        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        return DataLoader(dataset, batch_size=self.config["training"]["batch_size"], shuffle=True)

    def train(self) -> None:
        dataloader = self._build()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["training"]["lr"])
        criterion = nn.MSELoss()
        epochs = self.config["training"]["epochs"]

        self.model.train()
        for epoch in range(epochs):
            loss = self._epoch(dataloader, optimizer, criterion)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss:.6f}")

    def _epoch(self, dataloader, optimizer, criterion) -> float:
        total_loss = 0.0
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            optimizer.zero_grad()
            loss = criterion(self.model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)
        return total_loss / len(dataloader.dataset)

    def save_model(self) -> None:
        model_path = self.config["output"]["model_path"]
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved --> {model_path}")

    def run(self) -> None:
        self.train()
        self.save_model()
