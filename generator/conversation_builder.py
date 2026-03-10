from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConversationConfig:
    multi_turn: bool
    include_reasoning: bool


class ConversationBuilder:
    def __init__(self, system_prompt: str, cfg: ConversationConfig):
        self.system_prompt = system_prompt
        self.cfg = cfg

    def build_instruction_prompt(self, seed_text: str) -> str:
        return (
            f"<|system|>\n{self.system_prompt}\n\n"
            "You are generating high-quality synthetic user instructions for training.\n"
            "Respect this domain seed:\n"
            f"{seed_text}\n\n"
            "Create exactly one realistic user instruction."
        )

    def build_response_prompt(self, instruction: str, seed_text: str) -> str:
        reasoning_line = "Include concise step-by-step reasoning before the final answer." if self.cfg.include_reasoning else "Give a direct, accurate answer."
        return (
            f"<|system|>\n{self.system_prompt}\n\n"
            f"Domain seed:\n{seed_text}\n\n"
            f"<|user|>\n{instruction}\n\n"
            "<|assistant|>\n"
            f"{reasoning_line}"
        )

    def build_follow_up_prompt(self, instruction: str, response: str, seed_text: str) -> str:
        return (
            f"<|system|>\n{self.system_prompt}\n\n"
            f"Domain seed:\n{seed_text}\n\n"
            f"Prior user instruction: {instruction}\n"
            f"Prior assistant response: {response}\n\n"
            "Generate one realistic follow-up user instruction that deepens the same topic."
        )
