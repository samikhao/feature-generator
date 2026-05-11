import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import run_data_analysis_agent


def load_dataset(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("Agent evaluation dataset must be a JSON array")
    return data


def tool_recall(expected_tools: list[str], actual_tools: list[str]) -> float:
    expected = set(expected_tools)
    if not expected:
        return 1.0
    actual = set(actual_tools)
    return len(expected & actual) / len(expected)


def tool_precision(expected_tools: list[str], actual_tools: list[str]) -> float:
    if not actual_tools:
        return 0.0
    expected = set(expected_tools)
    actual = set(actual_tools)
    return len(expected & actual) / len(actual)


def keyword_recall(required_keywords: list[str], final_answer: str) -> float:
    if not required_keywords:
        return 1.0
    lowered = final_answer.lower()
    hits = sum(1 for keyword in required_keywords if keyword.lower() in lowered)
    return hits / len(required_keywords)


def policy_compliance(tool_calls: list[dict]) -> float:
    for call in tool_calls:
        arguments = call.get("arguments", {})
        file_path = arguments.get("file_path")
        if file_path and not str(file_path).startswith("data/"):
            return 0.0
    return 1.0


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def evaluate_case(example: dict) -> dict:
    run_result = run_data_analysis_agent(example["user_message"])
    tool_calls = [record.model_dump() for record in run_result.tool_calls]
    actual_tools = [call["name"] for call in tool_calls]
    metrics = {
        "tool_recall": round(tool_recall(example["expected_tools"], actual_tools), 4),
        "tool_precision": round(tool_precision(example["expected_tools"], actual_tools), 4),
        "keyword_recall": round(keyword_recall(example["required_keywords"], run_result.final_answer), 4),
        "policy_compliance": round(policy_compliance(tool_calls), 4),
        "success": 1.0,
    }
    return {
        "scenario_id": example["scenario_id"],
        "user_message": example["user_message"],
        "expected_tools": example["expected_tools"],
        "required_keywords": example["required_keywords"],
        "final_answer": run_result.final_answer,
        "tool_calls": tool_calls,
        "metrics": metrics,
    }


def aggregate(case_results: list[dict]) -> dict:
    return {
        "cases": len(case_results),
        "success_rate": round(mean([case["metrics"]["success"] for case in case_results]), 4),
        "mean_tool_recall": round(mean([case["metrics"]["tool_recall"] for case in case_results]), 4),
        "mean_tool_precision": round(mean([case["metrics"]["tool_precision"] for case in case_results]), 4),
        "mean_keyword_recall": round(mean([case["metrics"]["keyword_recall"] for case in case_results]), 4),
        "mean_policy_compliance": round(mean([case["metrics"]["policy_compliance"] for case in case_results]), 4),
    }


def build_summary_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_summary.json")


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def run_evaluation(dataset_path: Path, output_path: Path) -> dict:
    dataset = load_dataset(dataset_path)
    case_results = []
    for example in dataset:
        try:
            case_results.append(evaluate_case(example))
        except Exception as exc:
            case_results.append(
                {
                    "scenario_id": example["scenario_id"],
                    "user_message": example["user_message"],
                    "expected_tools": example["expected_tools"],
                    "required_keywords": example["required_keywords"],
                    "final_answer": "",
                    "tool_calls": [],
                    "metrics": {
                        "tool_recall": 0.0,
                        "tool_precision": 0.0,
                        "keyword_recall": 0.0,
                        "policy_compliance": 0.0,
                        "success": 0.0,
                    },
                    "error": str(exc),
                }
            )

    summary = aggregate(case_results)
    payload = {
        "dataset_path": str(dataset_path),
        "summary": summary,
        "cases": case_results,
    }
    write_json(output_path, payload)
    write_json(build_summary_path(output_path), summary)
    return payload


def main() -> None:
    output_path = PROJECT_ROOT / "artifacts" / "agent_eval_results.json"
    dataset_path = PROJECT_ROOT / "data" / "agent_eval_dataset.json"
    run_evaluation(dataset_path=dataset_path, output_path=output_path)
    print(output_path)


if __name__ == "__main__":
    main()
