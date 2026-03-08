import torch
import torch.nn as nn
from torch.utils.data import DataLoader as DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

from architecture.data import DataReader
from architecture.model import RegressionModel

class Analyzer:
    def __init__(self, csv_path, window_size=5, batch_size=16, epochs=50, lr=0.001, future=50, noise=0.01):
        self.data_loader = DataReader(csv_path, window_size)
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.future = future
        self.noise = noise
        self.model = RegressionModel(input_size=2)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    def prepare_data(self):
        X, y = self.data_loader.create_sequences()
        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def train(self):
        self.model.to(self.device)
        dataloader = self.prepare_data()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch_X.size(0)
            total_loss /= len(dataloader.dataset)
            if (epoch + 1) % 10 == 0: print(f"Epoch: {epoch+1}/{self.epochs} | Loss: {total_loss:.6f}")

    def predict(self, X):
        self.model.eval()
        predictions = []

        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float().to(self.device)
            pred = self.model(X_tensor).cpu().numpy()
            predictions.extend(pred)

            last_sequence = X[-1]
            for _ in range(self.future):
                seq_tensor = torch.from_numpy(last_sequence[np.newaxis, ...]).float().to(self.device)
                next_pred = self.model(seq_tensor).cpu().numpy()[0]
                next_pred += np.random.normal(scale=self.noise, size=next_pred.shape)
                predictions.append(next_pred)
                last_sequence = np.vstack([last_sequence[1:], next_pred])
        return self.data_loader.inverse_transform(np.array(predictions))

    def plot_graph(self, pred_data):
        timesteps = np.arange(len(pred_data))
        fig, ax1 = plt.subplots(figsize=(12, 5))

        line1, = ax1.plot(timesteps, pred_data[:, 0], label="CPU", color="red")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("CPU (%)", color="red")
        ax1.tick_params(axis="y", labelcolor="red")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Memory (MB)", color="blue")
        line2, = ax2.plot(timesteps, pred_data[:, 1], label="Memory", color="blue")
        ax2.tick_params(axis="y", labelcolor="blue")

        ax1.set_yticks(np.linspace(ax1.get_ylim()[0], ax1.get_ylim()[1], 5))
        ax2.set_yticks(np.linspace(ax2.get_ylim()[0], ax2.get_ylim()[1], 5))
        ax1.grid(linestyle="--")

        lines = [line1, line2]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels)