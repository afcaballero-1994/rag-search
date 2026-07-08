import argparse
import json
import os
from pathlib import Path
from typing import TypedDict
import string
from nltk.stem import PorterStemmer
from collections import defaultdict
from collections import Counter
import pickle
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
CACHE_PATH = os.path.join(PROJECT_ROOT, "cache")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")

class Movie(TypedDict):
    id: int
    title: str
    description: str

class InvertedIndex:
    index: dict[str, set[int]] = defaultdict(set)
    docmap: dict[int, Movie] = {}
    term_frequencies: dict[int, Counter] = defaultdict(Counter)


    def __add_document(self, doc_id: int, text: str) -> None:
        tokens: list[str] = tokenize_text(text)

        for token in set(tokens):
            self.index[token].add(doc_id)

        self.term_frequencies[doc_id].update(tokens)

    def get_documents(self, term: str) -> list[int]:
        result: list[int] = list(self.index[term])
        return sorted(result)

    def get_tf(self, doc_id: int, term: str) -> int:
        
        return self.term_frequencies[doc_id][term]

    def tokenize_term(self, term: str) -> str:
        token: list[str] = tokenize_text(term)
        if len(token) != 1:
            raise Exception("Too many tokens generated with term tokenizer")
        return token[0]

    def build(self) -> None:
        movies: list[Movie] = load_movies()

        for movie in movies:
            self.__add_document(movie["id"], f"{movie['title']} {movie["description"]}")
            self.docmap[movie["id"]] = movie

    def save(self) -> None:
        os.makedirs(CACHE_PATH, exist_ok=True)
        index_path: str = os.path.join(CACHE_PATH, "index.pkl")
        docmap_path: str = os.path.join(CACHE_PATH, "docmap.pkl")
        term_path: str = os.path.join(CACHE_PATH, "term_frequencies.pkl")
        
        with open(index_path, "wb") as f:
            pickle.dump(self.index, f)

        with open(docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)

        with open(term_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self) -> None:
        index_path: str = os.path.join(CACHE_PATH, "index.pkl")
        docmap_path: str = os.path.join(CACHE_PATH, "docmap.pkl")
        term_path: str = os.path.join(CACHE_PATH, "term_frequencies.pkl")

        try:
            with open(index_path, "rb") as f1:
                self.index = pickle.load(f1)
        except FileNotFoundError:
            print(f"Error: file {index_path} does not exist")
            sys.exit()

        try:
            with open(term_path, "rb") as f2:
                self.term_frequencies = pickle.load(f2)
        except FileNotFoundError:
            print(f"Error: file {term_path} does not exist")
            sys.exit()

        try:
            with open(docmap_path, "rb") as f:
                self.docmap = pickle.load(f)
        except FileNotFoundError:
            print(f"Error: file {docmap_path} does not exist")
            
def search_command(query: str, limit: int = 5):
    iidx = InvertedIndex()
    iidx.load()
    i: int = 0
    query_processed: list[str] = tokenize_text(query)
    results: list[int] = []

    for qsearch in query_processed:
        response: list[int] = iidx.get_documents(qsearch)
        if not response:
            continue
        else:
            results.extend(response)


    for key in results:
        movie = iidx.docmap[key]
        print(f"{i + 1}. ({movie["id"]}) {movie["title"]}")
        i += 1
        if i == limit:
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

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    subparsers.add_parser("build", help="Build the index reverse")
    tf_parser = subparsers.add_parser("tf", help="Show the term freqency")
    search_parser.add_argument("query", type=str, help="Search query")
    tf_parser.add_argument("doc_id", type=int, help="Doc ID document for frequency")
    tf_parser.add_argument("term", type=str, help="Term frequency arg")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            search_command(args.query)
        case "build":
            i = InvertedIndex()
            i.build()
            i.save()
        case "tf":
            i = InvertedIndex()
            i.load()
            term = i.tokenize_term(args.term)
            fq = i.get_tf(args.doc_id, term)
            print(f"The frequency of {args.term} is {fq}")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
