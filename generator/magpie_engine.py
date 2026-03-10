from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass


@dataclass
class GenerationParams:
    temperature: float
    top_p: float
    max_tokens: int
    batch_size: int
    max_parallel_requests: int
    repetition_penalty: float
    no_repeat_ngram_size: int


class MagpieEngine:
    def __init__(self, model_generate_fn, params: GenerationParams):
        self.model_generate_fn = model_generate_fn
        self.params = params

    def generate_texts(self, prompts: list[str]) -> list[str]:
        batches = [prompts[i : i + self.params.batch_size] for i in range(0, len(prompts), self.params.batch_size)]

        def run_batch(batch_prompts: list[str]) -> list[str]:
            return self.model_generate_fn(
                batch_prompts,
                temperature=self.params.temperature,
                top_p=self.params.top_p,
                max_tokens=self.params.max_tokens,
                repetition_penalty=self.params.repetition_penalty,
                no_repeat_ngram_size=self.params.no_repeat_ngram_size,
            )

        outputs: list[str] = []
        with ThreadPoolExecutor(max_workers=self.params.max_parallel_requests) as ex:
            for out in ex.map(run_batch, batches):
                outputs.extend(out)
        return outputs
