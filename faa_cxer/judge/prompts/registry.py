"""Registry of CxER judge prompt variants, plus per-model chat-template
wrapping. Kept as a small class (rather than the free-function + module
dict the original used) so a caller can register a custom prompt variant
at runtime without editing this file.
"""

from __future__ import annotations

from typing import Dict

from .base import BASE_PROMPT_CORE
from .compressed import COMPRESSED_COT_PROMPT
from .cot import COT_PROMPT_CORE
from .cot_v3 import COT_PROMPT_CORE as COT_V3_PROMPT_CORE
from .lenient import LENIENT_COT_PROMPT
from .no_examples import NO_EXAMPLES_COT_PROMPT

# Chat-template wrappers, keyed by a substring to match against the
# (lowercased) model name. Checked in order; first match wins.
_CHAT_TEMPLATES = [
    ("mistral", lambda core: f"[INST] {core} [/INST]"),
    ("gemma", lambda core: f"<start_of_turn>user\n{core}<end_of_turn>\n<start_of_turn>model\n"),
    ("llama", lambda core: f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{core}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"),
    ("qwen", lambda core: f"<|im_start|>user\n{core}<|im_end|>\n<|im_start|>assistant\n"),
    ("deepseek", lambda core: f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{core}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"),
]

DEFAULT_PROMPT_TYPE = "cot_v3"


class PromptRegistry:
    """Builds a full judge prompt (task template + chat wrapper) for a given
    (reference, transcription, model_name, prompt_type) combination.

    Prompt variants exist to support the ablation in the paper (does a more
    detailed chain-of-thought prompt change CxER?) — `cot_v3` is the one the
    main results use; the others are kept for that comparison.
    """

    def __init__(self):
        self._templates: Dict[str, str] = {
            "base": BASE_PROMPT_CORE,
            "cot": COT_PROMPT_CORE,
            "cot_v3": COT_V3_PROMPT_CORE,
            "compressed": COMPRESSED_COT_PROMPT,
            "no_examples": NO_EXAMPLES_COT_PROMPT,
            "lenient": LENIENT_COT_PROMPT,
        }

    def register(self, name: str, template: str) -> None:
        self._templates[name] = template

    def available(self):
        return list(self._templates)

    def build(self, reference: str, transcription: str, model_name: str, prompt_type: str = DEFAULT_PROMPT_TYPE) -> str:
        template = self._templates.get(prompt_type)
        if template is None:
            raise ValueError(f"Unknown prompt_type '{prompt_type}'. Available: {self.available()}")

        # .replace() rather than .format(): the templates contain literal
        # JSON braces in the "output this JSON" instructions, which would
        # collide with str.format()'s own brace syntax.
        core = template.replace("{reference}", reference.strip())
        core = core.replace("{transcription}", transcription.strip())

        model = model_name.lower()
        for needle, wrap in _CHAT_TEMPLATES:
            if needle in model:
                return wrap(core)
        return core
