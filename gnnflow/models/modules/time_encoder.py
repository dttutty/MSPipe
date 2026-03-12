import torch
import torch.nn as nn


class TimeEncoder(nn.Module):
    def __init__(self, time_dim: int):
        super().__init__()
        self.linear = nn.Linear(1, time_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(1)
        orig_shape = t.shape
        t = t.reshape(-1, 1)
        out = torch.cos(self.linear(t))
        return out.reshape(*orig_shape, -1)
