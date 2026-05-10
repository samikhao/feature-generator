from typing import Literal
from pydantic import BaseModel, Field


class ColumnInfo(BaseModel):
    name: str = Field(..., description="Column name")
    dtype: str = Field(
        ..., description="Column type, e.g. numeric, categorical, datetime"
    )
    description: str | None = Field(None, description="Optional column description")


class FeatureGenerationRequest(BaseModel):
    project_goal: str = Field(..., description="ML task goal")
    target_name: str = Field(..., description="Target column name")
    target_type: Literal["binary", "multiclass", "regression"]
    columns: list[ColumnInfo]
    dataset_context: str | None = Field(
        None,
        description="Additional context: domain, business meaning, data quality, etc.",
    )
    constraints: list[str] | None = Field(
        default_factory=list,
        description="Restrictions, e.g. avoid leakage, keep interpretability",
    )
    temperature: float = Field(0.2, ge=0.0, le=2.0)


class FeatureItem(BaseModel):
    feature_name: str
    source_columns: list[str]
    transformation: str
    description: str
    rationale: str
    priority: Literal["high", "medium", "low"]
    leakage_risk: Literal["low", "medium", "high"]


class FeatureGenerationResult(BaseModel):
    summary: str
    generated_features: list[FeatureItem]
    warnings: list[str]


class TaskUnderstanding(BaseModel):
    objective: str = Field(..., description="Restated modeling objective")
    target_name: str = Field(..., description="Prediction target")
    target_type: Literal["binary", "multiclass", "regression"]
    business_context: str = Field(..., description="Short domain context")
    success_criteria: list[str] = Field(
        ..., description="What good engineered features should achieve"
    )
    leakage_watchouts: list[str] = Field(
        ..., description="Potential leakage or temporal risks to watch"
    )


class CandidateFeature(BaseModel):
    feature_name: str
    source_columns: list[str]
    transformation: str
    description: str
    rationale: str
    priority: Literal["high", "medium", "low"]
    leakage_risk: Literal["low", "medium", "high"]
    keep: bool = Field(..., description="Whether this idea should be included")
    rejection_reason: str | None = Field(
        None,
        description="Why the feature was rejected or deprioritized",
    )


class FeatureGenerationPlan(BaseModel):
    task_understanding: TaskUnderstanding
    constraint_checklist: list[str] = Field(
        ..., description="Constraints and checks that must be respected"
    )
    candidate_features: list[CandidateFeature] = Field(
        ..., description="Generated and screened feature candidates"
    )
    final_response: FeatureGenerationResult


class UsageMetrics(BaseModel):
    model: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    total_tokens: int


class GenerationAudit(BaseModel):
    schema_version: str
    prompt_version: str
    task_summary: str
    applied_constraints: list[str]
    rejected_feature_ideas: list[str]
    usage: UsageMetrics


class FeatureGenerationResponse(FeatureGenerationResult):
    audit: GenerationAudit
