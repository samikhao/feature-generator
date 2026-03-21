import json
from fastapi import FastAPI, HTTPException
from src.schemas import (
    FeatureGenerationRequest,
    FeatureGenerationResponse,
)
from src.prompts import SYSTEM_PROMPT
from src.llm import generate_json


app = FastAPI(
    title="Feature Generator for ML",
    version="1.0.0",
    description="Prototype API for LLM-based feature generation",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate-features", response_model=FeatureGenerationResponse)
def generate_features(payload: FeatureGenerationRequest):
    user_prompt = build_user_prompt(payload)

    try:
        result = generate_json(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=payload.temperature,
        )
        return FeatureGenerationResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def build_user_prompt(payload: FeatureGenerationRequest) -> str:
    columns_json = json.dumps(
        [col.model_dump() for col in payload.columns],
        ensure_ascii=False,
        indent=2,
    )

    constraints = payload.constraints or []

    return f"""
Generate feature engineering ideas for the following ML problem.

Project goal:
{payload.project_goal}

Target:
- name: {payload.target_name}
- type: {payload.target_type}

Columns:
{columns_json}

Dataset context:
{payload.dataset_context or "Not provided"}

Constraints:
{json.dumps(constraints, ensure_ascii=False, indent=2)}

Return:
- short summary
- list of generated features
- warnings
"""
