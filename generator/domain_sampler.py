from __future__ import annotations

import random
from itertools import product
from typing import Any


class DomainSampler:
    def __init__(self, dataset_type: str, domain_structure: dict[str, list[str]], seed: int = 42) -> None:
        self.dataset_type = dataset_type
        self.domain_structure = domain_structure
        self.random = random.Random(seed)

    def sample_seed(self) -> dict[str, str]:
        return {key: self.random.choice(values) for key, values in self.domain_structure.items() if values}

    def exhaustive_combinations(self) -> list[dict[str, str]]:
        keys = list(self.domain_structure.keys())
        values = [self.domain_structure[key] for key in keys]
        combos = []
        for combo in product(*values):
            combos.append(dict(zip(keys, combo)))
        self.random.shuffle(combos)
        return combos

    def sample_batch(self, size: int) -> list[dict[str, str]]:
        combos = self.exhaustive_combinations()
        if not combos:
            return [self.sample_seed() for _ in range(size)]
        batch = []
        for idx in range(size):
            batch.append(combos[idx % len(combos)])
        self.random.shuffle(batch)
        return batch

    @staticmethod
    def format_seed(seed: dict[str, Any]) -> str:
        return "\n".join(f"- {key}: {value}" for key, value in seed.items())
