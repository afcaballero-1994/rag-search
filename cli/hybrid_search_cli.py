import argparse
import os
from pathlib import Path
import json
from openai import OpenAI
from dotenv import load_dotenv

from lib.hybrid_search import normalize_command
from lib.hybrid_search import HybridSearch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")

def load_movies() ->list[dict]:
    with open(DATA_PATH) as file:
        data = json.load(file)
    return data["movies"]

def main() -> None:
    load_dotenv()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
       raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    client = OpenAI(
       base_url="https://openrouter.ai/api/v1",
       api_key=api_key,
   )

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
        choices=["spell"],
        help="Query enhancement method",
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
            response = None

            if args.enhance == "spell":
                response = client.chat.completions.create(
                    model="openrouter/free",
                    messages = [
                        {
                            "role": "user",
                            "content": f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{args.query}"
"""
                        }
                    ]
                )

            if response is not None:
                query = response.choices[0].message.content
                print(f"Enhanced query: ({args.enhance}): '{args.query}' -> '{query}'")
            else:
                query = args.query

            if query is None:
                raise RuntimeError("Invalid query")
            
            results = mh.rrf_search(query, args.k, args.limit)
            
            for idx, doc in enumerate(results, 1):
                print(f"{idx}. {doc["title"]}\nRRF Score: {doc["rrf_score"]:.3f}")
                print(f"BM25 Rank: {doc["bm25_rank"]}, Semantic Rank: {doc["semantic_rank"]}")
                print(f"{doc["document"]}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
