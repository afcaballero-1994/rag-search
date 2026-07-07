import argparse
import json
import os
from pathlib import Path
from typing import TypedDict
import string
from nltk.stem import PorterStemmer


class Movie(TypedDict):
    id: int
    title: str
    description: str

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")



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
    query_processed: list[str] = tokenize_text(query)
    found: bool = False

    for movie in movies:
        if i == limit:
            break
        for word in query_processed:
            for wtitle in tokenize_text(movie["title"]):
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
def load_stopwords() ->list[str]:
    result: list[str] = []
    with open(STOPWORDS_PATH, "r", encoding="utf-8") as f:
        file_str = preprocess_text(f.read())
        result = file_str.splitlines()

    return result


def preprocess_text(input: str) -> str:
    result = input.lower()
    table = str.maketrans("", "", string.punctuation)
    result = result.translate(table)
    
    return result

STOPWORDS: list[str] = [preprocess_text(x) for x in load_stopwords()]

def tokenize_text(input: str) -> list[str]:
    tokens = preprocess_text(input).split()
    result: list[str] = []
    stemmer = PorterStemmer()

    for token in tokens:
        if token:
            result.append(token)
    result = list(filter(lambda x: x not in STOPWORDS, result))
    result = [stemmer.stem(tok) for tok in result]
    return result
if __name__ == "__main__":
    main()
