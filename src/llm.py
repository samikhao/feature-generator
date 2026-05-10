import json
from time import perf_counter

from openai import OpenAI

from src.config import (
    API_BASE,
    API_KEY,
    MODEL_NAME,
)
from src.schemas import FeatureGenerationPlan, UsageMetrics


client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE,
)


def _enforce_closed_objects(node: dict | list) -> None:
    if isinstance(node, dict):
        if node.get("type") == "object":
            node.setdefault("additionalProperties", False)
        for value in node.values():
            if isinstance(value, (dict, list)):
                _enforce_closed_objects(value)
    elif isinstance(node, list):
        for item in node:
            if isinstance(item, (dict, list)):
                _enforce_closed_objects(item)


def _build_response_schema() -> dict:
    schema = FeatureGenerationPlan.model_json_schema()
    _enforce_closed_objects(schema)
    return {
        "name": "feature_generation_plan",
        "strict": True,
        "schema": schema,
    }


def generate_plan(
    system_prompt: str,
    user_prompt: str,
    temperature: float,
) -> tuple[FeatureGenerationPlan, UsageMetrics]:
    started_at = perf_counter()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": _build_response_schema(),
        },
    )
    latency_ms = round((perf_counter() - started_at) * 1000, 2)

    content = response.choices[0].message.content
    if not content:
        raise ValueError("Model returned empty response")

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Model did not return valid JSON: {exc}\nRaw content: {content}"
        ) from exc

    plan = FeatureGenerationPlan.model_validate(parsed)
    usage = response.usage

    usage_metrics = UsageMetrics(
        model=MODEL_NAME,
        latency_ms=latency_ms,
        input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
        output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
        total_tokens=getattr(usage, "total_tokens", 0) if usage else 0,
    )
    return plan, usage_metrics
