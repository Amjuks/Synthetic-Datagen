from __future__ import annotations

import random
from typing import Any


class DomainSampler:
    def __init__(self, dataset_type: str, domain_structure: dict[str, list[str]] | None, seed: int | None = None) -> None:
        self.dataset_type = dataset_type
        self.domain_structure = domain_structure or {}
        self.random = random.Random(seed)

    def sample_seed(self) -> dict[str, str]:
        return {key: self.random.choice(values) for key, values in self.domain_structure.items() if values}

    def sample_batch(self, size: int) -> list[dict[str, str]]:
        if not self.domain_structure:
            return [{} for _ in range(size)]
        return [self.sample_seed() for _ in range(size)]

    @staticmethod
    def format_seed(seed: dict[str, Any]) -> str:
        if not seed:
            return ""
        return "\n".join(f"- {key}: {value}" for key, value in seed.items())
