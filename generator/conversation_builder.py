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
        if seed_text:
            guidance = f"Use this optional domain signal as loose guidance:\n{seed_text}\n\n"
        else:
            guidance = (
                "Choose language, task type, context, and difficulty naturally on your own.\n"
                "Vary them broadly across samples.\n\n"
            )
        return (
            f"System role: {self.system_prompt}\n\n"
            "Write one natural user message for a coding conversation.\n"
            f"{guidance}"
            "Make it realistic and human. It can be short or detailed depending on the situation.\n"
            "It must describe a concrete coding situation (bug, feature request, failing test, performance issue, API integration, refactor, etc.).\n"
            "Include at least one specific technical detail (language, stack component, error text, input/output, constraints, or behavior).\n"
            "Do NOT write assistant greetings like 'How can I help you'.\n"
            "Return ONLY the user message text. No preface, no separators, no explanation.\n"
            "Output:"
        )

    def build_response_prompt(self, instruction: str, seed_text: str) -> str:
        reasoning_line = (
            "First provide short reasoning, then write `Final Answer:` followed by the final answer."
            if self.cfg.include_reasoning
            else "Give a direct, accurate answer."
        )
        domain_section = f"Domain signal:\n{seed_text}\n\n" if seed_text else ""
        return (
            f"System role: {self.system_prompt}\n\n"
            "Task: Answer the following user instruction.\n"
            f"{domain_section}"
            f"User instruction:\n{instruction}\n\n"
            "Respond like a professional assistant with natural tone and appropriate depth.\n"
            "Simple requests should get concise answers; complex requests should include more detail.\n"
            "Give an actionable solution, not a request for more details.\n"
            "If needed, provide code snippets, exact commands, and verification steps.\n"
            f"Answer format: {reasoning_line}\n\n"
            "Output:"
        )

    def build_follow_up_prompt(self, history: list[tuple[str, str]], seed_text: str) -> str:
        formatted_history = []
        for idx, (instruction, response) in enumerate(history, start=1):
            formatted_history.append(f"Turn {idx} user instruction: {instruction}")
            formatted_history.append(f"Turn {idx} assistant response: {response}")
        history_text = "\n".join(formatted_history)
        domain_section = f"Domain signal:\n{seed_text}\n\n" if seed_text else ""
        return (
            f"System role: {self.system_prompt}\n\n"
            f"{domain_section}"
            f"Conversation so far:\n{history_text}\n\n"
            "Write the next user message in a natural human style.\n"
            "It must stay connected to the prior conversation and move the same task forward.\n"
            "Do NOT write assistant greetings or meta commentary.\n"
            "Return ONLY that user message. No explanation or formatting wrappers."
        )
