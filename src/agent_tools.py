from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from statistics import mean, median

from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
MAX_PREVIEW_ROWS = 10
MAX_PROFILE_ROWS = 10000


class ToolPolicyError(ValueError):
    pass


class DatasetPreviewInput(BaseModel):
    file_path: str = Field(..., description="Path to a local CSV file inside the data directory")
    preview_rows: int = Field(
        5,
        ge=1,
        le=MAX_PREVIEW_ROWS,
        description="Number of rows to return in the preview",
    )


class DatasetPreviewOutput(BaseModel):
    file_path: str
    row_count: int
    column_count: int
    columns: list[str]
    preview: list[dict[str, str]]


class NumericSummary(BaseModel):
    min: float
    max: float
    mean: float
    median: float


class ColumnProfile(BaseModel):
    name: str
    inferred_type: str
    non_null_count: int
    missing_count: int
    missing_rate: float
    unique_count: int
    top_values: list[dict[str, int]]
    numeric_summary: NumericSummary | None = None


class ColumnProfileInput(BaseModel):
    file_path: str = Field(..., description="Path to a local CSV file inside the data directory")
    top_values_limit: int = Field(
        5,
        ge=1,
        le=10,
        description="Maximum number of frequent values to include for each column",
    )


class ColumnProfileOutput(BaseModel):
    file_path: str
    row_count: int
    profiles: list[ColumnProfile]


class DataQualityInput(BaseModel):
    file_path: str = Field(..., description="Path to a local CSV file inside the data directory")


class DataQualityIssue(BaseModel):
    column: str
    severity: str
    message: str


class DataQualityOutput(BaseModel):
    file_path: str
    issues: list[DataQualityIssue]
    summary: list[str]


class TargetAnalysisInput(BaseModel):
    file_path: str = Field(..., description="Path to a local CSV file inside the data directory")
    target_name: str = Field(..., description="Name of the target column")


class TargetAnalysisOutput(BaseModel):
    file_path: str
    target_name: str
    inferred_target_type: str
    summary: dict[str, object]


def _ensure_allowed_csv_path(file_path: str) -> Path:
    path = (PROJECT_ROOT / file_path).resolve() if not Path(file_path).is_absolute() else Path(file_path).resolve()
    try:
        path.relative_to(DATA_ROOT.resolve())
    except ValueError as exc:
        raise ToolPolicyError(
            "Tool policy violation: only CSV files inside the data directory can be analyzed."
        ) from exc

    if path.suffix.lower() != ".csv":
        raise ToolPolicyError("Tool policy violation: only .csv files are allowed.")
    if not path.exists():
        raise ToolPolicyError(f"CSV file does not exist: {path}")
    return path


@lru_cache(maxsize=16)
def _read_csv(path_str: str) -> tuple[list[str], tuple[dict[str, str], ...]]:
    path = Path(path_str)
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {path}")
        rows = tuple({key: (value or "").strip() for key, value in row.items()} for row in reader)
    if len(rows) > MAX_PROFILE_ROWS:
        raise ToolPolicyError(
            f"Tool policy violation: dataset has {len(rows)} rows, which exceeds the analysis limit of {MAX_PROFILE_ROWS}."
        )
    return list(reader.fieldnames), rows


def _is_missing(value: str) -> bool:
    return value == ""


def _parse_float(value: str) -> float | None:
    if _is_missing(value):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_datetime(value: str) -> datetime | None:
    if _is_missing(value):
        return None
    for candidate in (
        value,
        value.replace("Z", "+00:00"),
        f"{value}T00:00:00",
    ):
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            continue
    return None


def _infer_column_type(values: list[str]) -> str:
    non_null_values = [value for value in values if not _is_missing(value)]
    if not non_null_values:
        return "unknown"

    lowered = {value.lower() for value in non_null_values}
    if lowered.issubset({"true", "false", "0", "1", "yes", "no"}):
        return "boolean"

    numeric_values = [_parse_float(value) for value in non_null_values]
    if all(value is not None for value in numeric_values):
        if all(float(value).is_integer() for value in numeric_values if value is not None):
            return "integer"
        return "numeric"

    datetime_values = [_parse_datetime(value) for value in non_null_values]
    if all(value is not None for value in datetime_values):
        return "datetime"

    unique_count = len(set(non_null_values))
    average_length = mean(len(value) for value in non_null_values)
    unique_ratio = unique_count / len(non_null_values)
    if average_length > 20 or unique_ratio > 0.8:
        return "text"
    return "categorical"


def _top_values(values: list[str], limit: int) -> list[dict[str, int]]:
    counts = Counter(value for value in values if not _is_missing(value))
    return [{key: value} for key, value in counts.most_common(limit)]


def load_dataset_preview(data: DatasetPreviewInput) -> DatasetPreviewOutput:
    path = _ensure_allowed_csv_path(data.file_path)
    headers, rows = _read_csv(str(path))
    preview = [dict(row) for row in rows[: data.preview_rows]]
    return DatasetPreviewOutput(
        file_path=str(path.relative_to(PROJECT_ROOT)),
        row_count=len(rows),
        column_count=len(headers),
        columns=headers,
        preview=preview,
    )


def profile_columns(data: ColumnProfileInput) -> ColumnProfileOutput:
    path = _ensure_allowed_csv_path(data.file_path)
    headers, rows = _read_csv(str(path))

    profiles: list[ColumnProfile] = []
    for header in headers:
        values = [row.get(header, "") for row in rows]
        missing_count = sum(1 for value in values if _is_missing(value))
        non_null_values = [value for value in values if not _is_missing(value)]
        inferred_type = _infer_column_type(values)
        numeric_summary = None

        if inferred_type in {"integer", "numeric"} and non_null_values:
            parsed_values = [float(value) for value in non_null_values]
            numeric_summary = NumericSummary(
                min=round(min(parsed_values), 4),
                max=round(max(parsed_values), 4),
                mean=round(mean(parsed_values), 4),
                median=round(median(parsed_values), 4),
            )

        profiles.append(
            ColumnProfile(
                name=header,
                inferred_type=inferred_type,
                non_null_count=len(non_null_values),
                missing_count=missing_count,
                missing_rate=round(missing_count / len(values), 4) if values else 0.0,
                unique_count=len(set(non_null_values)),
                top_values=_top_values(values, data.top_values_limit),
                numeric_summary=numeric_summary,
            )
        )

    return ColumnProfileOutput(
        file_path=str(path.relative_to(PROJECT_ROOT)),
        row_count=len(rows),
        profiles=profiles,
    )


def analyze_data_quality(data: DataQualityInput) -> DataQualityOutput:
    profile = profile_columns(ColumnProfileInput(file_path=data.file_path))
    issues: list[DataQualityIssue] = []
    summary: list[str] = []

    for column in profile.profiles:
        if column.missing_rate >= 0.3:
            issues.append(
                DataQualityIssue(
                    column=column.name,
                    severity="high",
                    message=f"High missing rate: {column.missing_rate:.2%}",
                )
            )
        elif column.missing_rate > 0:
            issues.append(
                DataQualityIssue(
                    column=column.name,
                    severity="medium",
                    message=f"Contains missing values: {column.missing_rate:.2%}",
                )
            )

        if column.unique_count <= 1:
            issues.append(
                DataQualityIssue(
                    column=column.name,
                    severity="medium",
                    message="Column is constant or nearly constant.",
                )
            )

        if column.inferred_type in {"categorical", "text"} and profile.row_count > 0:
            uniqueness_ratio = column.unique_count / max(column.non_null_count, 1)
            if uniqueness_ratio > 0.9:
                issues.append(
                    DataQualityIssue(
                        column=column.name,
                        severity="low",
                        message="Very high cardinality; may behave like an identifier.",
                    )
                )

        if column.numeric_summary is not None:
            spread = column.numeric_summary.max - column.numeric_summary.min
            if spread == 0:
                issues.append(
                    DataQualityIssue(
                        column=column.name,
                        severity="medium",
                        message="Numeric column has zero variance.",
                    )
                )

    if not issues:
        summary.append("No major data quality issues detected within the row limit.")
    else:
        high_count = sum(1 for issue in issues if issue.severity == "high")
        medium_count = sum(1 for issue in issues if issue.severity == "medium")
        summary.append(
            f"Detected {len(issues)} quality issues: {high_count} high severity and {medium_count} medium severity."
        )

    return DataQualityOutput(
        file_path=profile.file_path,
        issues=issues,
        summary=summary,
    )


def analyze_target_column(data: TargetAnalysisInput) -> TargetAnalysisOutput:
    path = _ensure_allowed_csv_path(data.file_path)
    headers, rows = _read_csv(str(path))
    if data.target_name not in headers:
        raise ValueError(f"Target column '{data.target_name}' was not found in the dataset.")

    values = [row.get(data.target_name, "") for row in rows if not _is_missing(row.get(data.target_name, ""))]
    if not values:
        raise ValueError(f"Target column '{data.target_name}' contains no non-empty values.")

    inferred_type = _infer_column_type(values)
    summary: dict[str, object]

    if inferred_type in {"boolean", "categorical"}:
        counts = Counter(values)
        if len(counts) == 2:
            inferred_target_type = "binary"
        else:
            inferred_target_type = "multiclass"
        summary = {
            "class_counts": dict(counts),
            "class_ratio": {
                key: round(value / len(values), 4) for key, value in counts.items()
            },
        }
    elif inferred_type in {"integer", "numeric"}:
        numeric_values = [float(value) for value in values]
        distinct_values = len(set(values))
        if distinct_values == 2:
            inferred_target_type = "binary"
            counts = Counter(values)
            summary = {
                "class_counts": dict(counts),
                "class_ratio": {
                    key: round(value / len(values), 4) for key, value in counts.items()
                },
            }
        else:
            inferred_target_type = "regression"
            summary = {
                "min": round(min(numeric_values), 4),
                "max": round(max(numeric_values), 4),
                "mean": round(mean(numeric_values), 4),
                "median": round(median(numeric_values), 4),
                "distinct_values": distinct_values,
            }
    else:
        inferred_target_type = "text_like"
        summary = {
            "unique_values": len(set(values)),
            "sample_values": values[:5],
        }

    return TargetAnalysisOutput(
        file_path=str(path.relative_to(PROJECT_ROOT)),
        target_name=data.target_name,
        inferred_target_type=inferred_target_type,
        summary=summary,
    )


class ToolSpec(BaseModel):
    name: str
    description: str
    input_model_name: str
    output_model_name: str


class ToolCallRecord(BaseModel):
    name: str
    arguments: dict[str, object]
    output: dict[str, object]


TOOL_REGISTRY = {
    "load_dataset_preview": (DatasetPreviewInput, DatasetPreviewOutput, load_dataset_preview),
    "profile_columns": (ColumnProfileInput, ColumnProfileOutput, profile_columns),
    "analyze_data_quality": (DataQualityInput, DataQualityOutput, analyze_data_quality),
    "analyze_target_column": (TargetAnalysisInput, TargetAnalysisOutput, analyze_target_column),
}


def build_openai_tool_specs() -> list[dict[str, object]]:
    specs = []
    for name, (input_model, _, _) in TOOL_REGISTRY.items():
        specs.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": _tool_description(name),
                    "parameters": input_model.model_json_schema(),
                },
            }
        )
    return specs


def _tool_description(name: str) -> str:
    descriptions = {
        "load_dataset_preview": "Load a local CSV dataset preview, column list, and row count.",
        "profile_columns": "Inspect actual values in each column and compute column profiles and numeric summaries.",
        "analyze_data_quality": "Detect missing values, constant columns, high-cardinality columns, and other quality issues.",
        "analyze_target_column": "Inspect the actual target column values and infer whether the target behaves like binary, multiclass, or regression.",
    }
    return descriptions[name]


def execute_tool_call(name: str, arguments_json: str) -> ToolCallRecord:
    if name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {name}")

    input_model, _, handler = TOOL_REGISTRY[name]
    arguments = json.loads(arguments_json)
    validated_input = input_model.model_validate(arguments)
    result = handler(validated_input)
    return ToolCallRecord(
        name=name,
        arguments=validated_input.model_dump(),
        output=result.model_dump(),
    )
