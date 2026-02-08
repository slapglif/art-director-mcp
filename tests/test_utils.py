from __future__ import annotations

import base64

from art_director.utils import (
    b64_to_image_bytes,
    estimate_generation_cost,
    estimate_pipeline_cost,
    image_bytes_to_b64,
    repair_json,
)


def test_image_bytes_roundtrip() -> None:
    original = b"\x89PNG\r\n\x1a\nfake_image_data"
    b64 = image_bytes_to_b64(original, "image/png")
    assert b64.startswith("data:image/png;base64,")
    decoded = b64_to_image_bytes(b64)
    assert decoded == original


def test_b64_without_prefix() -> None:
    data = b"hello world"
    raw_b64 = base64.b64encode(data).decode()
    decoded = b64_to_image_bytes(raw_b64)
    assert decoded == data


def test_repair_json_clean() -> None:
    result = repair_json('{"verdict": "pass", "score": 8.5}')
    assert result is not None
    assert result["verdict"] == "pass"


def test_repair_json_markdown_block() -> None:
    raw = '```json\n{"verdict": "fail", "score": 3.0}\n```'
    result = repair_json(raw)
    assert result is not None
    assert result["verdict"] == "fail"


def test_repair_json_single_quotes() -> None:
    raw = "{'verdict': 'pass', 'score': 9}"
    result = repair_json(raw)
    assert result is not None
    assert result["score"] == 9


def test_repair_json_trailing_comma() -> None:
    raw = '{"verdict": "pass", "score": 7,}'
    result = repair_json(raw)
    assert result is not None
    assert result["verdict"] == "pass"


def test_repair_json_garbage() -> None:
    result = repair_json("this is not json at all")
    assert result is None


def test_repair_json_prose_then_json() -> None:
    raw = (
        "The user wants to generate an image of a cat. Let me analyze the request carefully.\n\n"
        "Based on the requirements, I'll select a photorealistic model.\n\n"
        '{"intent_category": "photorealistic", "selected_model_id": "black-forest-labs/FLUX.1-dev", '
        '"prompt_optimized": "a fluffy orange cat sitting on a windowsill", "score": 8.5}'
    )
    result = repair_json(raw)
    assert result is not None
    assert result["intent_category"] == "photorealistic"
    assert result["score"] == 8.5


def test_repair_json_prose_with_braces_then_json() -> None:
    raw = (
        "I need to check {text_rendering} and {photorealism} capabilities.\n\n"
        "Here is the plan:\n"
        '{"selected_model_id": "stabilityai/stable-diffusion-xl-base-1.0", "verdict": "pass"}'
    )
    result = repair_json(raw)
    assert result is not None
    assert result["selected_model_id"] == "stabilityai/stable-diffusion-xl-base-1.0"


def test_estimate_generation_cost_known_model() -> None:
    cost = estimate_generation_cost("black-forest-labs/FLUX.1-dev")
    assert cost == 0.03


def test_estimate_generation_cost_unknown_model() -> None:
    cost = estimate_generation_cost("unknown/model")
    assert cost == 0.02


def test_estimate_pipeline_cost() -> None:
    cost = estimate_pipeline_cost("black-forest-labs/FLUX.1-dev", max_retries=3)
    assert cost > 0.03
