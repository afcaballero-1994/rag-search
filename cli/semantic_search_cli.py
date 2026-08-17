import argparse
import json
import os
from pathlib import Path

from keyword_search_cli import Movie
from lib import semantic_search

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")

def load_movies() ->list[Movie]:
    with open(DATA_PATH) as file:
        data = json.load(file)
    return data["movies"]

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic search CLI")
    subparsers = parser.add_subparsers(dest="commands", help="available commands")

    verify_parser = subparsers.add_parser("verify", help="Verify model loaded")

    embed_text_parser = subparsers.add_parser("embed_text", help="Generate embedding")
    embed_text_parser.add_argument("text", type=str, help="Text used to generate emb")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verify")

    embed_query = subparsers.add_parser("embed_query", help="Generate embedding query")
    embed_query.add_argument("query", type=str, help="Query to embbed")
    
    args = parser.parse_args()

    

    match args.commands:
        case "verify":
            semantic_search.verify_model()
        case "embed_text":
            semantic_search.embed_text(args.text)
        case "verify_embeddings":
            movies = load_movies()
            m = semantic_search.SemanticSearch()
            m.load_or_create_embeddings(movies)

            print(f"Number of docs: {len(m.documents)}")
            print(
                f"Embeddings shape: {m.embeddings.shape[0]} vectors in {m.embeddings.shape[1]} dimensions"
            )

        case "embed_query":
            semantic_search.embed_query(args.query)
            
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

