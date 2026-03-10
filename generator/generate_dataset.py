from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import re

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
    random_seed: int | None = None
    checkpoint_every: int = 100


class DatasetGenerator:
    def __init__(self, model_generate_fn, logger):
        self.model_generate_fn = model_generate_fn
        self.logger = logger
        self.random = random.Random()

    def run(self, cfg: DatasetRunConfig, output_path: str) -> pd.DataFrame:
        conv_cfg = ConversationConfig(
            multi_turn=bool(cfg.dataset_structure.get("multi_turn", False)),
            include_reasoning=bool(cfg.dataset_structure.get("include_reasoning", False)),
        )
        self.random = random.Random(cfg.random_seed)
        builder = ConversationBuilder(cfg.system_prompt, conv_cfg)
        sampler = DomainSampler(cfg.dataset_type, cfg.domain_structure, seed=cfg.random_seed)
        seeds = sampler.sample_batch(cfg.samples)

        params = GenerationParams(
            temperature=float(cfg.generation.get("temperature", 0.8)),
            top_p=float(cfg.generation.get("top_p", 0.95)),
            max_tokens=int(cfg.generation.get("max_tokens", 512)),
            batch_size=int(cfg.generation.get("batch_size", 8)),
            max_parallel_requests=int(cfg.generation.get("max_parallel_requests", 2)),
            repetition_penalty=float(cfg.generation.get("repetition_penalty", 1.1)),
            no_repeat_ngram_size=int(cfg.generation.get("no_repeat_ngram_size", 3)),
        )
        engine = MagpieEngine(self.model_generate_fn, params)
        max_retries = max(1, int(cfg.generation.get("max_retries", 5)))

        rows: list[dict] = []
        start_idx = self._restore_checkpoint(output_path, rows)

        for i in tqdm(range(start_idx, cfg.samples), desc="Generating"):
            seed = seeds[i]
            seed_text = sampler.format_seed(seed)
            turns = self._choose_turn_count(cfg.dataset_structure, conv_cfg.multi_turn)
            conversation: list[tuple[str, str]] = []

            instruction = self._generate_valid_instruction(engine, builder, seed_text, max_retries)
            response = self._generate_valid_response(engine, builder, instruction, seed_text, max_retries)
            conversation.append((instruction, response))

            for _ in range(2, turns + 1):
                next_instruction = self._generate_valid_followup_instruction(
                    engine, builder, conversation, seed_text, max_retries
                )
                next_response = self._generate_valid_response(
                    engine, builder, next_instruction, seed_text, max_retries
                )
                conversation.append((next_instruction, next_response))

            row = {**seed, "turn_count": turns}
            if conv_cfg.multi_turn:
                for idx, (inst, resp) in enumerate(conversation, start=1):
                    row[f"instruction_{idx}"] = inst
                    if conv_cfg.include_reasoning:
                        reasoning, final_response = self._split_reasoning_response(resp)
                        row[f"reasoning_{idx}"] = reasoning
                        row[f"response_{idx}"] = final_response
                    else:
                        row[f"response_{idx}"] = resp
            else:
                row["instruction"] = conversation[0][0]
                if conv_cfg.include_reasoning:
                    reasoning, final_response = self._split_reasoning_response(conversation[0][1])
                    row["reasoning"] = reasoning
                    row["response"] = final_response
                else:
                    row["response"] = conversation[0][1]

            rows.append(row)

            if (i + 1) % cfg.checkpoint_every == 0:
                self.logger.info("Checkpoint at sample %s", i + 1)
                self._write_checkpoint(output_path, rows)

        df = pd.DataFrame(rows)
        safe_write_csv(df, output_path)
        self._checkpoint_path(output_path).unlink(missing_ok=True)
        return df

    def _gen_one(self, engine: MagpieEngine, prompt: str) -> str:
        outputs = engine.generate_texts([prompt])
        return outputs[0].strip() if outputs else ""

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
            cp_df = pd.read_csv(cp_path, engine="python")
            rows.extend(cp_df.to_dict(orient="records"))
            self.logger.info("Resuming from checkpoint with %s rows", len(cp_df))
            return len(cp_df)
        return 0

    def _write_checkpoint(self, output_path: str, rows: list[dict]) -> None:
        cp_df = pd.DataFrame(rows)
        safe_write_csv(cp_df, self._checkpoint_path(output_path))

    def _choose_turn_count(self, dataset_structure: dict, multi_turn: bool) -> int:
        if not multi_turn:
            return 1
        min_turns = int(dataset_structure.get("min_turns", 2))
        max_turns = int(dataset_structure.get("max_turns", min_turns))
        if min_turns < 2:
            min_turns = 2
        if max_turns < min_turns:
            max_turns = min_turns
        return self.random.randint(min_turns, max_turns)

    def _extract_user_instruction(self, text: str) -> str:
        cleaned = text.replace("\r\n", "\n").strip()
        if not cleaned:
            return cleaned

        # Remove markdown fences and common wrappers.
        cleaned = cleaned.replace("```", "").strip()
        cleaned = cleaned.strip("\"' ")

        # Prefer content inside divider blocks if present.
        if "---" in cleaned:
            parts = [p.strip() for p in cleaned.split("---") if p.strip()]
            if parts:
                cleaned = parts[0] if len(parts) == 1 else parts[1]

        lines = [ln.strip(" -*\t") for ln in cleaned.splitlines() if ln.strip()]
        meta_prefixes = (
            "sure",
            "here",
            "this question",
            "this prompt",
            "explanation",
            "note:",
            "output:",
        )
        lines = [ln for ln in lines if not ln.lower().startswith(meta_prefixes)]
        if not lines:
            lines = [cleaned]

        for ln in lines:
            if "?" in ln:
                return ln[:400].strip()
        return lines[0][:400].strip()

    def _generate_valid_instruction(
        self, engine: MagpieEngine, builder: ConversationBuilder, seed_text: str, max_retries: int
    ) -> str:
        best = ""
        for _ in range(max_retries):
            candidate = self._extract_user_instruction(self._gen_one(engine, builder.build_instruction_prompt(seed_text)))
            if candidate and not best:
                best = candidate
            if self._is_valid_instruction(candidate):
                return candidate
        raise RuntimeError(f"Failed to generate a valid instruction after {max_retries} attempts. Last={best[:160]!r}")

    def _generate_valid_followup_instruction(
        self,
        engine: MagpieEngine,
        builder: ConversationBuilder,
        conversation: list[tuple[str, str]],
        seed_text: str,
        max_retries: int,
    ) -> str:
        best = ""
        for _ in range(max_retries):
            prompt = builder.build_follow_up_prompt(conversation, seed_text)
            candidate = self._extract_user_instruction(self._gen_one(engine, prompt))
            if candidate and not best:
                best = candidate
            if self._is_valid_instruction(candidate):
                return candidate
        raise RuntimeError(
            f"Failed to generate a valid follow-up instruction after {max_retries} attempts. Last={best[:160]!r}"
        )

    def _generate_valid_response(
        self,
        engine: MagpieEngine,
        builder: ConversationBuilder,
        instruction: str,
        seed_text: str,
        max_retries: int,
    ) -> str:
        best = ""
        for _ in range(max_retries):
            candidate = self._gen_one(engine, builder.build_response_prompt(instruction, seed_text))
            if candidate and not best:
                best = candidate
            if self._is_valid_response(candidate):
                return candidate
        raise RuntimeError(f"Failed to generate a valid response after {max_retries} attempts. Last={best[:160]!r}")

    def _is_valid_instruction(self, text: str) -> bool:
        if not text or len(text) < 20:
            return False
        lowered = text.lower()
        banned = [
            "how may i assist you",
            "how can i help you",
            "feel free to ask",
            "let me know if",
            "i can assist",
        ]
        if any(b in lowered for b in banned):
            return False
        # Require at least one coding signal and one concrete context/detail signal.
        coding_signals = [
            "python", "javascript", "typescript", "java", "c++", "go", "rust", "sql",
            "api", "endpoint", "query", "database", "function", "class", "test", "bug",
            "exception", "error", "stack trace", "latency", "memory", "timeout", "ci", "docker",
        ]
        detail_signals = [":", "http", "traceback", "500", "404", "ms", "input", "output", "expected", "actual", "?"]
        if not any(s in lowered for s in coding_signals):
            return False
        if not any(s in lowered for s in detail_signals):
            return False
        return True

    def _is_valid_response(self, text: str) -> bool:
        if not text or len(text) < 40:
            return False
        lowered = text.lower()
        banned = [
            "how can i help",
            "please provide more details",
            "feel free to ask",
            "let me know what you need",
        ]
        if any(b in lowered for b in banned):
            return False
        actionable_signals = [
            "1.", "2.", "```", "try", "use", "update", "set", "run", "query", "function",
            "because", "fix", "example", "test", "verify", "command",
        ]
        return any(s in lowered for s in actionable_signals)
