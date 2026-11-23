"""Escolha do dispositivo: GPU NVIDIA (cuda), GPU Apple Silicon (mps) ou CPU."""
import os
import torch


def get_device(name=None):
    name = name or os.environ.get("HTR_DEVICE")
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_type(device):
    """Precisão mista só na GPU NVIDIA; no Mac (MPS) e na CPU o treino roda em float32."""
    return "cuda" if device.type == "cuda" else "cpu"
