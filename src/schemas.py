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


class FeatureGenerationResponse(BaseModel):
    summary: str
    generated_features: list[FeatureItem]
    warnings: list[str]
