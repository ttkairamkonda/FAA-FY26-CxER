import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faa_cxer.judge.prompts import PromptRegistry


def test_available_prompt_types():
    registry = PromptRegistry()
    for name in ("base", "cot", "cot_v3", "compressed", "no_examples", "lenient"):
        assert name in registry.available()


def test_build_substitutes_reference_and_transcription():
    registry = PromptRegistry()
    prompt = registry.build("united two three four", "united two tree", "gpt-4", prompt_type="base")
    assert "united two three four" in prompt
    assert "united two tree" in prompt
    assert "{reference}" not in prompt
    assert "{transcription}" not in prompt


def test_llama_chat_template_wraps_prompt():
    registry = PromptRegistry()
    prompt = registry.build("ref", "hyp", "meta-llama/Llama-3.3-70B-Instruct", prompt_type="base")
    assert prompt.startswith("<|begin_of_text|>")
    assert "<|eot_id|>" in prompt


def test_unknown_model_gets_no_wrapper():
    registry = PromptRegistry()
    prompt = registry.build("ref", "hyp", "some-unlisted-model", prompt_type="base")
    assert not prompt.startswith("<|")
    assert not prompt.startswith("[INST]")


def test_unknown_prompt_type_raises():
    registry = PromptRegistry()
    with pytest.raises(ValueError):
        registry.build("ref", "hyp", "gpt-4", prompt_type="does_not_exist")


def test_register_custom_prompt():
    registry = PromptRegistry()
    registry.register("my_variant", "Compare {reference} to {transcription}.")
    prompt = registry.build("A", "B", "gpt-4", prompt_type="my_variant")
    assert prompt == "Compare A to B."
