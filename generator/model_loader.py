from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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

        def _generate(
            prompts: list[str],
            temperature: float,
            top_p: float,
            max_tokens: int,
            repetition_penalty: float = 1.0,
            no_repeat_ngram_size: int = 0,
        ) -> list[str]:
            params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                repetition_penalty=repetition_penalty,
            )
            outputs = llm.generate(prompts, params)
            return [o.outputs[0].text.strip() for o in outputs]

        return GenerationBackend(backend_name="vllm", generator=_generate)

    def _load_transformers(self) -> GenerationBackend:
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        model_kwargs: dict[str, Any] = {
            "device_map": "auto" if self.device == "cuda" else None,
            "low_cpu_mem_usage": True,
        }

        # On CUDA, prefer quantized loading when bitsandbytes is available.
        # This helps fit 8B-class models on limited VRAM (e.g., P100 16GB)
        # and avoids heavy CPU offloading.
        quantized_loaded = False
        if self.device == "cuda" and has_package("bitsandbytes"):
            try:
                from transformers import BitsAndBytesConfig

                bnb_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                q_kwargs = dict(model_kwargs)
                q_kwargs["quantization_config"] = bnb_cfg
                q_kwargs["torch_dtype"] = torch.float16
                model = AutoModelForCausalLM.from_pretrained(self.model_name, **q_kwargs)
                quantized_loaded = True
            except Exception:
                quantized_loaded = False

        if not quantized_loaded:
            # Compatibility across transformers versions: newer favors `dtype`,
            # older accepts `torch_dtype`.
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    dtype=dtype,
                    **model_kwargs,
                )
            except TypeError:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=dtype,
                    **model_kwargs,
                )
        model.eval()
        run_device = "cuda" if self.device == "cuda" and torch.cuda.is_available() else "cpu"

        def _encode_prompt(prompt: str) -> dict[str, torch.Tensor]:
            if getattr(tokenizer, "chat_template", None):
                messages = [{"role": "user", "content": prompt}]
                encoded = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                if isinstance(encoded, torch.Tensor):
                    input_ids = encoded
                    attention_mask = torch.ones_like(input_ids)
                    return {"input_ids": input_ids, "attention_mask": attention_mask}

                # Some transformers builds return BatchEncoding here.
                input_ids = encoded["input_ids"]
                attention_mask = encoded.get("attention_mask")
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids)
                return {"input_ids": input_ids, "attention_mask": attention_mask}

            encoded = tokenizer(prompt, return_tensors="pt")
            input_ids = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        def _generate(
            prompts: list[str],
            temperature: float,
            top_p: float,
            max_tokens: int,
            repetition_penalty: float = 1.0,
            no_repeat_ngram_size: int = 0,
        ) -> list[str]:
            outputs: list[str] = []
            with torch.inference_mode():
                for prompt in prompts:
                    encoded = _encode_prompt(prompt)
                    encoded = {k: v.to(run_device) for k, v in encoded.items()}
                    input_len = int(encoded["input_ids"].shape[-1])

                    generated = model.generate(
                        **encoded,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        max_new_tokens=max_tokens,
                        min_new_tokens=16,
                        repetition_penalty=repetition_penalty,
                        no_repeat_ngram_size=no_repeat_ngram_size,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                    completion_ids = generated[0][input_len:]
                    completion = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
                    outputs.append(completion)
            return outputs

        return GenerationBackend(backend_name="transformers", generator=_generate)
