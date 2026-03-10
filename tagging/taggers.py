from __future__ import annotations

from dataclasses import dataclass
import re


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
        self.alias_to_lang = {
            "py": "python",
            "python3": "python",
            "js": "javascript",
            "node": "javascript",
            "nodejs": "javascript",
            "ts": "typescript",
            "tsx": "typescript",
            "golang": "go",
            "rs": "rust",
            "cpp": "c++",
            "cxx": "c++",
            "cc": "c++",
            "csharp": "c#",
            "cs": "c#",
            "ps1": "powershell",
            "pwsh": "powershell",
            "shell": "bash",
            "sh": "bash",
            "postgres": "sql",
            "mysql": "sql",
            "sqlite": "sql",
            "mssql": "sql",
        }

    def tag_row(self, row: dict) -> dict:
        instruction = str(row.get("instruction") or row.get("instruction_1") or "").lower()
        response = str(row.get("response") or row.get("response_1") or "").lower()
        combined_text = f"{instruction}\n{response}"
        row_language = self._normalize_lang(str(row.get("languages", "")).lower().strip())
        row_topic = str(row.get("topics", "")).lower().strip()
        row_level = str(row.get("difficulty_levels", "")).lower().strip()

        if row_language in self.defs.language and row_language != "unknown":
            language = row_language
        else:
            language = self._detect_language(combined_text)

        math_keywords = [
            "equation",
            "integral",
            "proof",
            "algebra",
            "geometry",
            "calculus",
            "trigonometry",
            "probability",
            "statistics",
            "number theory",
        ]
        if row_topic:
            category = "math"
        elif language != "unknown":
            category = "coding"
        else:
            category = "math" if any(x in instruction for x in math_keywords) else "general"

        if row_level in self.defs.difficulty:
            difficulty = row_level
        elif any(k in instruction for k in ["advanced", "optimize", "concurrency", "proof", "derive"]):
            difficulty = "advanced"
        elif any(k in instruction for k in ["beginner", "elementary", "intro"]):
            difficulty = "beginner"
        else:
            difficulty = "intermediate"
        quality = "high" if len(response) > 60 else "medium"
        safety = "safe" if not any(k in response for k in ["malware", "exploit", "harm"]) else "review"

        return {
            "tag_category": category if category in self.defs.category else "general",
            "tag_difficulty": difficulty if difficulty in self.defs.difficulty else self.defs.difficulty[0],
            "tag_language": language,
            "tag_quality": quality if quality in self.defs.quality else self.defs.quality[0],
            "tag_safety": safety if safety in self.defs.safety else self.defs.safety[0],
        }

    def _normalize_lang(self, raw: str) -> str:
        if not raw:
            return "unknown"
        raw = raw.strip().lower()
        return self.alias_to_lang.get(raw, raw)

    def _detect_language(self, text: str) -> str:
        lowered = text.lower()

        # Prefer fenced code block hints when present.
        fence_matches = re.findall(r"```([a-zA-Z0-9#+\-.]+)", lowered)
        for hint in fence_matches:
            normalized = self._normalize_lang(hint)
            if normalized in self.defs.language and normalized != "unknown":
                return normalized

        # Then alias and canonical name matches in the full text.
        for alias, canonical in self.alias_to_lang.items():
            if canonical in self.defs.language and self._contains_language(lowered, alias):
                return canonical
        for lang in self.defs.language:
            if lang != "unknown" and self._contains_language(lowered, lang):
                return lang
        return "unknown"

    def _contains_language(self, text: str, lang: str) -> bool:
        if lang in {"c++", "c#"}:
            return lang in text
        pattern = rf"\b{re.escape(lang)}\b"
        return bool(re.search(pattern, text))
