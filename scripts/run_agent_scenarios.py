import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import run_data_analysis_agent


SCENARIOS = [
    {
        "scenario_id": "customer_churn_numeric_review",
        "user_message": (
            "Изучи файл data/customer_churn_sample.csv. "
            "Посмотри на реальные значения в столбцах, оцени числовые признаки, "
            "распределение таргета churn и коротко скажи, какие фичи стоит попробовать."
        ),
        "expected_tools": [
            "load_dataset_preview",
            "profile_columns",
            "analyze_target_column",
        ],
    },
    {
        "scenario_id": "loan_risk_quality_review",
        "user_message": (
            "Проанализируй data/loan_risk_sample.csv. "
            "Найди проблемы качества данных, посмотри на target defaulted и "
            "сделай краткие выводы по полезным направлениям для feature engineering."
        ),
        "expected_tools": [
            "analyze_data_quality",
            "analyze_target_column",
        ],
    },
]


def evaluate_tool_usage(expected_tools: list[str], actual_tools: list[str]) -> dict:
    expected = set(expected_tools)
    actual = set(actual_tools)
    return {
        "all_expected_tools_used": expected.issubset(actual),
        "missing_tools": sorted(expected - actual),
        "unexpected_tools": sorted(actual - expected),
    }


def main() -> None:
    results = []
    for scenario in SCENARIOS:
        run_result = run_data_analysis_agent(scenario["user_message"])
        used_tools = [tool_call.name for tool_call in run_result.tool_calls]
        results.append(
            {
                "scenario_id": scenario["scenario_id"],
                "user_message": scenario["user_message"],
                "final_answer": run_result.final_answer,
                "tool_calls": [record.model_dump() for record in run_result.tool_calls],
                "tool_usage_check": evaluate_tool_usage(
                    scenario["expected_tools"],
                    used_tools,
                ),
            }
        )

    output_path = PROJECT_ROOT / "artifacts" / "agent_scenarios_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    print(output_path)


if __name__ == "__main__":
    main()
