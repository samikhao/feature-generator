import argparse
import csv
import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.main import app


def load_dataset(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Проверочный набор данных должен быть JSON-массивом")

    return data


def run_evaluation(dataset_path: Path, output_path: Path) -> None:
    dataset = load_dataset(dataset_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with TestClient(app) as client, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["case_id", "request", "result", "status_code", "error"],
        )
        writer.writeheader()

        for index, example in enumerate(dataset, start=1):
            case_id = example.get("case_id", f"case_{index}")
            request_body = example.get("request", example)

            response = client.post("/generate-features", json=request_body)

            try:
                response_body = response.json()
            except json.JSONDecodeError:
                response_body = {"raw_text": response.text}

            writer.writerow(
                {
                    "case_id": case_id,
                    "request": json.dumps(request_body, ensure_ascii=False),
                    "result": json.dumps(response_body, ensure_ascii=False),
                    "status_code": response.status_code,
                    "error": response_body.get("detail", "")
                    if isinstance(response_body, dict)
                    else "",
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Запустить проверочный набор данных через API генерации признаков"
    )
    parser.add_argument(
        "--dataset",
        default="data/eval_dataset.json",
        help="Путь к проверочному набору данных в формате JSON",
    )
    parser.add_argument(
        "--output",
        default="artifacts/eval_results.csv",
        help="Путь к выходной сводной таблице в формате CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_evaluation(Path(args.dataset), Path(args.output))


if __name__ == "__main__":
    main()
