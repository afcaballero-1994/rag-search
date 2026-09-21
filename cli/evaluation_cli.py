import argparse
import json, os
from pathlib import Path

from hybrid_search_cli import HybridSearch, load_movies

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "golden_dataset.json")

def main() -> None:
    parser = argparse.ArgumentParser(description="Search evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)"
    )

    args = parser.parse_args()

    limit = args.limit

    with open(GOLDEN_DATA_PATH, "r") as f:
        golden_data = json.load(f)

    dataset: list[dict] = golden_data["test_cases"]

    movies = load_movies()
    msearch = HybridSearch(movies)

    for entry in dataset:
        query: str = entry["query"]
        relevant_docs: list[str] = entry["relevant_docs"]

        print(f"Current query: {query}")
        results = msearch.rrf_search(query, 60, limit)

        total_retrieved = len(results)
        relevant_retrieved = 0

        for doc in results:
            if doc["title"] in relevant_docs:
                relevant_retrieved += 1

        titles = [p["title"] for p in results]
        
        precision = relevant_retrieved / total_retrieved
        print(f"k={limit}")
        print(f"-Query: {query}\n -Precision@{limit}: {precision:.4f}")
        print(f" - Retrieved: {', '.join(titles)}\n - Relevant: {', '.join(relevant_docs)}")

    
    
if __name__ == "__main__":
    main()
