from __future__ import annotations

from dataclasses import dataclass, asdict

import torch


@dataclass
class HardwareInfo:
    cuda_available: bool
    gpu_count: int
    gpu_names: list[str]
    device: str



def detect_hardware() -> HardwareInfo:
    cuda_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if cuda_available else 0
    gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
    device = "cuda" if cuda_available else "cpu"
    return HardwareInfo(
        cuda_available=cuda_available,
        gpu_count=gpu_count,
        gpu_names=gpu_names,
        device=device,
    )


def hardware_to_dict(info: HardwareInfo) -> dict:
    return asdict(info)
