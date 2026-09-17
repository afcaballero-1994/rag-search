import argparse
import os
from pathlib import Path
import json


from lib.hybrid_search import normalize_command
from lib.hybrid_search import HybridSearch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")

def load_movies() ->list[dict]:
    with open(DATA_PATH) as file:
        data = json.load(file)
    return data["movies"]

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available Commands")

    normalize_parser = subparser.add_parser("normalize", help="Normalize list values")
    normalize_parser.add_argument("values", type=float, nargs="*", help="Values to normal")

    weighted_search_parser = subparser.add_parser("weighted-search", help="Wie search")
    weighted_search_parser.add_argument("query", type=str, help="terms to search")
    weighted_search_parser.add_argument("--alpha",
                                        nargs="?", type=float,
                                        default=0.5
                                        )
    weighted_search_parser.add_argument(
        "--limit",
        nargs="?", type=int,
        default=5
    )
    
    args = parser.parse_args()

    match args.command:
        case "normalize":
            result = normalize_command(args.values)
            for r in result:
                print(f"* {r:.4f}")
        case "weighted-search":
            movies = load_movies()
            mh = HybridSearch(movies)
            results = mh.weighted_search(args.query, args.alpha, args.limit)

            for idx, doc in enumerate(results, 1):
                print(f"{idx}. {doc["title"]}")
                print(f"Hybrid Score: {doc["hybrid_score"]:.3f}")
                print(f"BM25: {doc["bm25_score"]:.3f}, Semantic: {doc["semantic_score"]:.3f}")
                print(f"{doc["description"]}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
