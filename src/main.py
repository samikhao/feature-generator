from fastapi import FastAPI, HTTPException

from src.llm import generate_plan
from src.prompts import PROMPT_SPECS, build_user_prompt, get_prompt_spec
from src.schemas import (
    FeatureGenerationRequest,
    FeatureGenerationResponse,
    GenerationAudit,
)


app = FastAPI(
    title="Feature Generator for ML",
    version="1.0.0",
    description="Prototype API for LLM-based feature generation",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/prompt-versions")
def list_prompt_versions():
    return [
        {
            "version": spec.version,
            "title": spec.title,
            "description": spec.description,
        }
        for spec in PROMPT_SPECS
    ]


@app.post("/generate-features", response_model=FeatureGenerationResponse)
def generate_features(
    payload: FeatureGenerationRequest,
    prompt_version: str | None = None,
):
    try:
        return run_feature_generation(payload, prompt_version=prompt_version)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def run_feature_generation(
    payload: FeatureGenerationRequest,
    prompt_version: str | None = None,
) -> FeatureGenerationResponse:
    prompt_spec = get_prompt_spec(prompt_version)
    user_prompt = build_user_prompt(payload, prompt_version=prompt_spec.version)
    plan, usage_metrics = generate_plan(
        system_prompt=prompt_spec.system_prompt,
        user_prompt=user_prompt,
        temperature=payload.temperature,
    )

    rejected_feature_ideas = [
        candidate.feature_name
        for candidate in plan.candidate_features
        if not candidate.keep
    ]
    response_payload = plan.final_response.model_dump()
    response_payload["audit"] = GenerationAudit(
        schema_version="feature_generation_plan.v1",
        prompt_version=prompt_spec.version,
        task_summary=plan.task_understanding.objective,
        applied_constraints=plan.constraint_checklist,
        rejected_feature_ideas=rejected_feature_ideas,
        usage=usage_metrics,
    ).model_dump()
    return FeatureGenerationResponse.model_validate(response_payload)
