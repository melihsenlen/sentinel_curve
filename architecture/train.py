import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from architecture.data import DataReader
from architecture.model import RegressionModel
from architecture.config import load_config, data, training, output

class Trainer:
    def __init__(self):
        self.config = load_config()
        csv_path = data(self.config)["csv_path"]
        print(f"Reading: {csv_path}")

        self.reader = DataReader(csv_path, data(self.config)["window_size"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RegressionModel(input_size=2).to(self.device)

    def _build(self) -> DataLoader:
        X, y = self.reader.create_sequences()
        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        return DataLoader(dataset, batch_size=training(self.config)["batch_size"], shuffle=True)

    def train(self) -> None:
        dataloader = self._build()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=training(self.config)["lr"])
        criterion = nn.MSELoss()
        epochs = training(self.config)["epochs"]

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
        model_path = output(self.config)["model_path"]
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved --> {model_path}")

    def run(self) -> None:
        self.train()
        self.save_model()
