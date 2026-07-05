import argparse
import json
import os
from pathlib import Path
from typing import TypedDict
import string

class Movie(TypedDict):
    id: int
    title: str
    description: str

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            search_command(args.query)

        case _:
            parser.print_help()


def search_command(query: str, limit: int = 5):
    movies = load_movies()
    i: int = 0
    query_processed: list[str] = preprocess_text(query)
    found: bool = False

    for movie in movies:
        if i == limit:
            break
        for word in query_processed:
            for wtitle in preprocess_text(movie["title"]):
                if word in wtitle:
                    print(f"{i + 1}. {movie["title"]}")
                    i += 1
                    found = True
                    break
            if found:
                found = False
                break
            
    

def load_movies() ->list[Movie]:
    with open(DATA_PATH) as file:
        data = json.load(file)
    return data["movies"]

def preprocess_text(input: str) -> list[str]:
    result = input.lower()
    to_remove: str = string.punctuation
    table = str.maketrans("", "", to_remove)
    result = result.translate(table)
    result = list(filter(None, result.split()))
    return result

if __name__ == "__main__":
    main()
