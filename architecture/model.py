import torch
import torch.nn as nn


class RegressionModel(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 32, num_layers: int = 1):
        super(RegressionModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim) # Output size, same as input size (cpu, memory)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x