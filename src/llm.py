import json
from openai import OpenAI
from src.config import API_KEY, API_BASE, MODEL_NAME


client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE,
)


FEATURE_RESPONSE_SCHEMA = {
    "name": "feature_generation_response",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "summary": {"type": "string"},
            "generated_features": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "feature_name": {"type": "string"},
                        "source_columns": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "transformation": {"type": "string"},
                        "description": {"type": "string"},
                        "rationale": {"type": "string"},
                        "priority": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                        },
                        "leakage_risk": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                        },
                    },
                    "required": [
                        "feature_name",
                        "source_columns",
                        "transformation",
                        "description",
                        "rationale",
                        "priority",
                        "leakage_risk",
                    ],
                },
            },
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["summary", "generated_features", "warnings"],
    },
}


def generate_json(system_prompt: str, user_prompt: str, temperature: float) -> dict:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": FEATURE_RESPONSE_SCHEMA,
        },
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError("Model returned empty response")

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Model did not return valid JSON: {e}\nRaw content: {content}"
        )
