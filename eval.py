import json
from typing import Any, Dict, List

from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision

from src.agent import respond_to_query


def load_golden_dataset(path: str = "data/golden_dataset.json") -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def run_evaluation():
    dataset = load_golden_dataset()
    results = []

    for item in dataset:
        answer = respond_to_query(item["question"])
        results.append(
            {
                "question": item["question"],
                "answer": answer,
                "ground_truth": item["answer"],
                "contexts": [item["context"]],
            }
        )

    score = evaluate(results, metrics=[answer_relevancy, context_precision])
    print(score)


if __name__ == "__main__":
    run_evaluation()
