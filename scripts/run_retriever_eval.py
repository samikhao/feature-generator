import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retriever import retrieve_knowledge
from src.schemas import FeatureGenerationRequest


def load_dataset(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("Retriever evaluation dataset must be a JSON array")
    return data


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for title in top_k if title in relevant)
    return hits / len(top_k)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for title in top_k if title in relevant)
    return hits / len(relevant)


def hit_rate_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if any(title in relevant for title in retrieved[:k]) else 0.0


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for index, title in enumerate(retrieved, start=1):
        if title in relevant:
            return 1.0 / index
    return 0.0


def average_precision(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for index, title in enumerate(retrieved[:k], start=1):
        if title in relevant:
            hits += 1
            precision_sum += hits / index

    if hits == 0:
        return 0.0
    return precision_sum / len(relevant)


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def evaluate_case(example: dict, top_k: int) -> dict:
    request = FeatureGenerationRequest.model_validate(example["request"])
    relevant_titles = set(example["relevant_titles"])
    retrieval = retrieve_knowledge(request, top_k=top_k)
    retrieved_titles = [item.title for item in retrieval.items]

    metrics = {
        "precision_at_k": round(precision_at_k(retrieved_titles, relevant_titles, top_k), 4),
        "recall_at_k": round(recall_at_k(retrieved_titles, relevant_titles, top_k), 4),
        "hit_rate_at_k": round(hit_rate_at_k(retrieved_titles, relevant_titles, top_k), 4),
        "mrr": round(reciprocal_rank(retrieved_titles, relevant_titles), 4),
        "average_precision_at_k": round(
            average_precision(retrieved_titles, relevant_titles, top_k),
            4,
        ),
    }

    return {
        "case_id": example["case_id"],
        "description": example.get("description", ""),
        "query": retrieval.query,
        "relevant_titles": example["relevant_titles"],
        "retrieved_titles": retrieved_titles,
        "retrieved_items": [item.model_dump() for item in retrieval.items],
        "metrics": metrics,
    }


def aggregate_results(case_results: list[dict], top_k: int) -> dict:
    return {
        "cases": len(case_results),
        "top_k": top_k,
        "mean_precision_at_k": round(mean([case["metrics"]["precision_at_k"] for case in case_results]), 4),
        "mean_recall_at_k": round(mean([case["metrics"]["recall_at_k"] for case in case_results]), 4),
        "hit_rate_at_k": round(mean([case["metrics"]["hit_rate_at_k"] for case in case_results]), 4),
        "mrr": round(mean([case["metrics"]["mrr"] for case in case_results]), 4),
        "map_at_k": round(mean([case["metrics"]["average_precision_at_k"] for case in case_results]), 4),
    }


def build_summary_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_summary.json")


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def run_evaluation(dataset_path: Path, output_path: Path, top_k: int) -> dict:
    dataset = load_dataset(dataset_path)
    case_results = [evaluate_case(example, top_k=top_k) for example in dataset]
    summary = aggregate_results(case_results, top_k=top_k)

    output_payload = {
        "dataset_path": str(dataset_path),
        "summary": summary,
        "cases": case_results,
    }
    write_json(output_path, output_payload)
    write_json(build_summary_path(output_path), summary)
    return output_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run retrieval evaluation for the local feature-engineering RAG retriever"
    )
    parser.add_argument(
        "--dataset",
        default="data/retrieval_eval_dataset.json",
        help="Path to the retrieval evaluation dataset",
    )
    parser.add_argument(
        "--output",
        default="artifacts/retriever_eval_results.json",
        help="Path to the detailed evaluation results",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of retrieved sections to evaluate",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_evaluation(
        dataset_path=Path(args.dataset),
        output_path=Path(args.output),
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
