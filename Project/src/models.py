"""
models.py
---------
Contains three neural network architectures for comparison:

1. CNN1D   - Fast, local pattern extraction
2. LSTMseq - Recurrent, good for long-term temporal memory
3. TCN     - Temporal Convolutional Network, strong for sequence modeling
"""

import torch
import torch.nn as nn


# ------------------------------------------------------------
# 1. CNN1D model
# ------------------------------------------------------------
class CNN1D(nn.Module):
    """Simple 1D CNN for time-series fault detection."""
    def __init__(self, in_ch: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        # x: [B, T, C] → permute to [B, C, T]
        x = x.permute(0, 2, 1)
        z = self.net(x).squeeze(-1)
        return self.fc(z)


# ------------------------------------------------------------
# 2. LSTM model
# ------------------------------------------------------------
class LSTMseq(nn.Module):
    """LSTM-based classifier for longer temporal memory."""
    def __init__(self, in_ch: int, n_classes: int, hidden=64, layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_ch, hidden_size=hidden,
                            num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x):
        # x: [B, T, C]
        output, (h, _) = self.lstm(x)
        return self.fc(h[-1])  # last layer hidden state


# ------------------------------------------------------------
# 3. Temporal Convolutional Network (TCN)
# ------------------------------------------------------------
class TCN(nn.Module):
    """TCN with dilated convolutions for long-range dependencies."""
    def __init__(self, in_ch: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        z = self.net(x).squeeze(-1)
        return self.fc(z)
