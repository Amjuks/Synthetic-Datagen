from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from utils.environment_check import has_package


@dataclass
class GenerationBackend:
    backend_name: str
    generator: Any


class ModelLoader:
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device

    def load(self) -> GenerationBackend:
        if has_package("vllm"):
            return self._load_vllm()
        return self._load_transformers()

    def _load_vllm(self) -> GenerationBackend:
        from vllm import LLM, SamplingParams

        llm = LLM(model=self.model_name, tensor_parallel_size=max(1, torch.cuda.device_count()))

        def _generate(prompts: list[str], temperature: float, top_p: float, max_tokens: int) -> list[str]:
            params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
            outputs = llm.generate(prompts, params)
            return [o.outputs[0].text.strip() for o in outputs]

        return GenerationBackend(backend_name="vllm", generator=_generate)

    def _load_transformers(self) -> GenerationBackend:
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )

        text_gen = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if self.device == "cuda" else -1,
        )

        def _generate(prompts: list[str], temperature: float, top_p: float, max_tokens: int) -> list[str]:
            outs = text_gen(
                prompts,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_tokens,
                return_full_text=False,
                batch_size=min(8, len(prompts)),
            )
            return [out[0]["generated_text"].strip() for out in outs]

        return GenerationBackend(backend_name="transformers", generator=_generate)
