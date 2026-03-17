import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def apply_temperature(logits, temperature: float):
    if temperature is None or temperature == 1.0:
        return logits
    return logits / temperature


def fit_temperature_binary(logits, targets, t_min=0.5, t_max=5.0, steps=60):
    temps = np.linspace(t_min, t_max, steps)
    best_t = 1.0
    best_loss = 1e9
    logits_np = logits.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    for t in temps:
        z = logits_np / t
        loss = np.mean(np.logaddexp(0.0, z) - targets_np * z)
        if loss < best_loss:
            best_loss = loss
            best_t = float(t)
    return best_t


def save_temperature(path: str, temperature: float):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"temperature": temperature}, indent=2))


def load_temperature(path: str) -> float:
    data = json.loads(Path(path).read_text())
    return float(data.get("temperature", 1.0))
