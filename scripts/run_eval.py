import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.main import run_feature_generation
from src.prompts import PROMPT_SPECS
from src.schemas import FeatureGenerationRequest, FeatureGenerationResponse


def load_dataset(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("Проверочный набор данных должен быть JSON-массивом")

    return data


def build_summary_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_summary.json")


def compute_quality_score(
    request: FeatureGenerationRequest,
    response: FeatureGenerationResponse,
) -> tuple[float, dict]:
    max_score = 6.0
    allowed_columns = {column.name for column in request.columns}
    generated_features = response.generated_features

    completeness = 1.0 if response.summary and generated_features else 0.0

    source_column_hits = 0
    target_leakage_violations = 0
    for feature in generated_features:
        if set(feature.source_columns).issubset(allowed_columns):
            source_column_hits += 1
        if request.target_name in feature.source_columns:
            target_leakage_violations += 1

    column_validity = (
        source_column_hits / len(generated_features) if generated_features else 0.0
    )
    leakage_safety = 0.0 if target_leakage_violations else 1.0

    rationale_coverage = (
        sum(1 for feature in generated_features if feature.rationale.strip())
        / len(generated_features)
        if generated_features
        else 0.0
    )

    warning_signal = 1.0 if response.warnings else 0.5

    feature_count = len(generated_features)
    if 2 <= feature_count <= 6:
        feature_count_score = 1.0
    elif feature_count == 1 or 7 <= feature_count <= 8:
        feature_count_score = 0.5
    else:
        feature_count_score = 0.0

    raw_score = (
        completeness
        + column_validity
        + leakage_safety
        + rationale_coverage
        + warning_signal
        + feature_count_score
    )
    normalized_score = round(raw_score / max_score, 4)
    diagnostics = {
        "completeness": completeness,
        "column_validity": round(column_validity, 4),
        "leakage_safety": leakage_safety,
        "rationale_coverage": round(rationale_coverage, 4),
        "warning_signal": warning_signal,
        "feature_count_score": feature_count_score,
        "generated_feature_count": feature_count,
        "target_leakage_violations": target_leakage_violations,
    }
    return normalized_score, diagnostics


def write_summary(summary_path: Path, summary: dict) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)


def run_evaluation(dataset_path: Path, output_path: Path) -> None:
    dataset = load_dataset(dataset_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict] = {}

    with output_path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "case_id",
                "prompt_version",
                "prompt_title",
                "status",
                "quality_score",
                "quality_details",
                "latency_ms",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "request",
                "result",
                "error",
            ],
        )
        writer.writeheader()

        for prompt_spec in PROMPT_SPECS:
            per_prompt_scores: list[float] = []
            per_prompt_latency: list[float] = []

            for index, example in enumerate(dataset, start=1):
                case_id = example.get("case_id", f"case_{index}")
                request_body = example.get("request", example)

                try:
                    request = FeatureGenerationRequest.model_validate(request_body)
                    response = run_feature_generation(
                        request,
                        prompt_version=prompt_spec.version,
                    )
                    validated_response = FeatureGenerationResponse.model_validate(
                        response.model_dump()
                    )
                    quality_score, quality_details = compute_quality_score(
                        request=request,
                        response=validated_response,
                    )
                    usage = validated_response.audit.usage
                    status = "ok"
                    error = ""

                    per_prompt_scores.append(quality_score)
                    per_prompt_latency.append(usage.latency_ms)

                    result_payload = validated_response.model_dump()
                except Exception as exc:
                    quality_score = 0.0
                    quality_details = {"error": str(exc)}
                    usage = None
                    status = "error"
                    error = str(exc)
                    result_payload = {}

                writer.writerow(
                    {
                        "case_id": case_id,
                        "prompt_version": prompt_spec.version,
                        "prompt_title": prompt_spec.title,
                        "status": status,
                        "quality_score": quality_score,
                        "quality_details": json.dumps(
                            quality_details,
                            ensure_ascii=False,
                        ),
                        "latency_ms": usage.latency_ms if usage else "",
                        "input_tokens": usage.input_tokens if usage else "",
                        "output_tokens": usage.output_tokens if usage else "",
                        "total_tokens": usage.total_tokens if usage else "",
                        "request": json.dumps(request_body, ensure_ascii=False),
                        "result": json.dumps(result_payload, ensure_ascii=False),
                        "error": error,
                    }
                )

            summary[prompt_spec.version] = {
                "title": prompt_spec.title,
                "description": prompt_spec.description,
                "cases": len(dataset),
                "avg_quality_score": round(
                    sum(per_prompt_scores) / len(per_prompt_scores), 4
                )
                if per_prompt_scores
                else 0.0,
                "avg_latency_ms": round(
                    sum(per_prompt_latency) / len(per_prompt_latency), 2
                )
                if per_prompt_latency
                else 0.0,
            }

    write_summary(build_summary_path(output_path), summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Запустить проверочный набор данных через все версии промпта и "
            "сравнить качество и скорость"
        )
    )
    parser.add_argument(
        "--dataset",
        default="data/eval_dataset.json",
        help="Путь к проверочному набору данных в формате JSON",
    )
    parser.add_argument(
        "--output",
        default="artifacts/prompt_eval_results.csv",
        help="Путь к выходной сводной таблице в формате CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_evaluation(Path(args.dataset), Path(args.output))


if __name__ == "__main__":
    main()
