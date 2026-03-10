from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from generator.conversation_builder import ConversationBuilder, ConversationConfig
from generator.domain_sampler import DomainSampler
from generator.magpie_engine import GenerationParams, MagpieEngine
from utils.io_utils import safe_write_csv


@dataclass
class DatasetRunConfig:
    dataset_type: str
    system_prompt: str
    domain_structure: dict
    dataset_structure: dict
    generation: dict
    samples: int
    checkpoint_every: int = 100


class DatasetGenerator:
    def __init__(self, model_generate_fn, logger):
        self.model_generate_fn = model_generate_fn
        self.logger = logger

    def run(self, cfg: DatasetRunConfig, output_path: str) -> pd.DataFrame:
        conv_cfg = ConversationConfig(
            multi_turn=bool(cfg.dataset_structure.get("multi_turn", False)),
            include_reasoning=bool(cfg.dataset_structure.get("include_reasoning", False)),
        )
        builder = ConversationBuilder(cfg.system_prompt, conv_cfg)
        sampler = DomainSampler(cfg.dataset_type, cfg.domain_structure)
        seeds = sampler.sample_batch(cfg.samples)

        params = GenerationParams(
            temperature=float(cfg.generation.get("temperature", 0.8)),
            top_p=float(cfg.generation.get("top_p", 0.95)),
            max_tokens=int(cfg.generation.get("max_tokens", 512)),
            batch_size=int(cfg.generation.get("batch_size", 8)),
            max_parallel_requests=int(cfg.generation.get("max_parallel_requests", 2)),
        )
        engine = MagpieEngine(self.model_generate_fn, params)

        rows: list[dict] = []
        start_idx = self._restore_checkpoint(output_path, rows)

        for i in tqdm(range(start_idx, cfg.samples), desc="Generating"):
            seed = seeds[i]
            seed_text = sampler.format_seed(seed)

            instruction = engine.generate_texts([builder.build_instruction_prompt(seed_text)])[0]
            response = engine.generate_texts([builder.build_response_prompt(instruction, seed_text)])[0]

            row = {"instruction": instruction, "response": response, **seed}

            if conv_cfg.include_reasoning:
                reasoning, final_response = self._split_reasoning_response(response)
                row["reasoning"] = reasoning
                row["response"] = final_response

            if conv_cfg.multi_turn:
                instruction_2 = engine.generate_texts([builder.build_follow_up_prompt(instruction, response, seed_text)])[0]
                response_2 = engine.generate_texts([builder.build_response_prompt(instruction_2, seed_text)])[0]
                row["instruction_1"] = instruction
                row["response_1"] = response
                row["instruction_2"] = instruction_2
                row["response_2"] = response_2
                row.pop("instruction", None)
                row.pop("response", None)

            rows.append(row)

            if (i + 1) % cfg.checkpoint_every == 0:
                self.logger.info("Checkpoint at sample %s", i + 1)
                self._write_checkpoint(output_path, rows)

        df = pd.DataFrame(rows)
        safe_write_csv(df, output_path)
        self._checkpoint_path(output_path).unlink(missing_ok=True)
        return df

    def _split_reasoning_response(self, response: str) -> tuple[str, str]:
        marker = "Final Answer:"
        if marker in response:
            reasoning, final = response.split(marker, 1)
            return reasoning.strip(), final.strip()
        return "", response.strip()

    def _checkpoint_path(self, output_path: str) -> Path:
        return Path(output_path).with_suffix(".checkpoint.csv")

    def _restore_checkpoint(self, output_path: str, rows: list[dict]) -> int:
        cp_path = self._checkpoint_path(output_path)
        if cp_path.exists():
            cp_df = pd.read_csv(cp_path)
            rows.extend(cp_df.to_dict(orient="records"))
            self.logger.info("Resuming from checkpoint with %s rows", len(cp_df))
            return len(cp_df)
        return 0

    def _write_checkpoint(self, output_path: str, rows: list[dict]) -> None:
        cp_df = pd.DataFrame(rows)
        safe_write_csv(cp_df, self._checkpoint_path(output_path))
