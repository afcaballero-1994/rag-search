import argparse
import os, logging
from pathlib import Path
import json

from lib.hybrid_search import normalize_command
from lib.hybrid_search import HybridSearch
from lib.hybrid_search import load_model, MODEL

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")

def load_movies() ->list[dict]:
    with open(DATA_PATH) as file:
        data = json.load(file)
    return data["movies"]

def get_prompt_evaluation(query: str, results: list[dict]) -> str:
    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{results}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers other than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""
    return prompt

def main() -> None:
    logging.basicConfig(filename="rff_search.log", level=logging.INFO)

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

    rrf_search_parser = subparser.add_parser("rrf-search", help="Search using rrf")
    rrf_search_parser.add_argument("query", type=str, help="Query search")
    rrf_search_parser.add_argument(
        "-k", type=int, nargs="?", default=60
    )
    rrf_search_parser.add_argument(
        "--limit", type=int, nargs="?", default=5
    )
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )

    rrf_search_parser.add_argument(
        "--rerank-method",
        type=str,
        nargs="?",
        choices=["individual", "batch", "cross_encoder"],
        help="Rerank method to be used"
    )

    rrf_search_parser.add_argument(
        "--evaluate",
        type=bool,
        action=argparse.BooleanOptionalAction,
        help="Add evaluation step made by LLM"
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
        case "rrf-search":
            movies = load_movies()
            mh = HybridSearch(movies)

            limit = args.limit
            if args.rerank_method is not None:
                limit *= 5
            
            results = mh.rrf_search(args.query, args.k, limit, args.enhance, args.rerank_method)
            
            for idx, doc in enumerate(results[:args.limit], 1):
                print(f"{idx}. {doc["title"]}")
                if args.rerank_method is not None:
                    if args.rerank_method == "individual":
                        print(f"- Re-rank Score: {doc["rerank_score"]}/10")
                    if args.rerank_method == "batch":
                        print(f"- Re-rank Score: {doc["rerank_score"]}")
                    if args.rerank_method == "cross_encoder":
                        print(f"- Cross Encoder Score: {doc["cross_encoder_score"]:.3f}")
                print(f"- RRF Score: {doc["rrf_score"]:.3f}")
                print(f"- BM25 Rank: {doc["bm25_rank"]}, Semantic Rank: {doc["semantic_rank"]}")
                print(f"-- {doc["document"]}...")

            if args.evaluate:
                prompt = get_prompt_evaluation(args.query, results)
                client = load_model()
                response = client.chat.completions.create(
                    model=MODEL, messages=[{"role": "user", "content": prompt}]
                )

                data: str | None = response.choices[0].message.content
                if data is None:
                    raise RuntimeError("No response received model evaluation step")
                scores = json.loads(data)
                
                for jdx, score in enumerate(scores):
                    print(f"{jdx + 1}. {results[jdx]["title"]}: {score}/3")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
