import torch
import torch.nn as nn
from models.quantum import QuantumLayer

class VQCOnly(nn.Module):
    """Config B: simple CNN encoder + VQC, NO ViT backbone.
    Expected to perform poorly - proves the backbone is essential."""

    def __init__(self, cfg):
        super().__init__()
        nc = cfg["data"]["num_classes"]
        nq = cfg["quantum"]["n_qubits"]
        nl = cfg["quantum"]["n_layers"]
        std = cfg["quantum"]["init_std"]

        # Simple CNN encoder (replaces ViT)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # 224->112
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 112->56
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),                    # 56->1x1
            nn.Flatten(),                               # -> 32
            nn.Linear(32, nq),                          # -> 4
        )
        self.vqc = QuantumLayer(n_qubits=nq, n_layers=nl, init_std=std)
        self.head = nn.Linear(nq, nc)

    def forward(self, x):
        features = self.encoder(x)
        quantum_out = self.vqc(features)
        return self.head(quantum_out)
