from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TagDefinitions:
    category: list[str]
    difficulty: list[str]
    language: list[str]
    quality: list[str]
    safety: list[str]


class RuleBasedTagger:
    def __init__(self, defs: TagDefinitions):
        self.defs = defs

    def tag_row(self, row: dict) -> dict:
        instruction = str(row.get("instruction") or row.get("instruction_1") or "").lower()
        response = str(row.get("response") or row.get("response_1") or "").lower()

        language = next((lang for lang in self.defs.language if lang in instruction), "unknown")
        category = "coding" if language != "unknown" else ("math" if any(x in instruction for x in ["equation", "integral", "proof"]) else "general")
        difficulty = "advanced" if any(k in instruction for k in ["optimize", "concurrency", "proof", "derive"]) else "intermediate"
        quality = "high" if len(response) > 60 else "medium"
        safety = "safe" if not any(k in response for k in ["malware", "exploit", "harm"]) else "review"

        return {
            "tag_category": category if category in self.defs.category else "general",
            "tag_difficulty": difficulty if difficulty in self.defs.difficulty else self.defs.difficulty[0],
            "tag_language": language,
            "tag_quality": quality if quality in self.defs.quality else self.defs.quality[0],
            "tag_safety": safety if safety in self.defs.safety else self.defs.safety[0],
        }
