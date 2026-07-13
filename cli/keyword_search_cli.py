import argparse
import json
import os
from pathlib import Path
from typing import TypedDict
import string
from nltk.stem import PorterStemmer
from collections import (
    defaultdict,
    Counter
)

import pickle
import sys
import math

class Movie(TypedDict):
    id: int
    title: str
    description: str

class InvertedIndex:
    index: dict[str, set[int]] = defaultdict(set)
    docmap: dict[int, Movie] = {}
    stemmer: PorterStemmer
    table = str.maketrans("", "", string.punctuation)
    term_frequencies: dict[int, Counter] = defaultdict(Counter)
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
    CACHE_PATH = os.path.join(PROJECT_ROOT, "cache")
    STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
    STOPWORDS: set[str]

    def __init__(self, load_cache: bool = True):
        self.STOPWORDS = set(self.preprocess_text(x) for x in self.load_stopwords())
        self.stemmer = PorterStemmer()
        if load_cache:
            self.load()


    def __add_document(self, doc_id: int, text: str) -> None:
        tokens: list[str] = self.tokenize_text(text)

        for token in set(tokens):
            self.index[token].add(doc_id)

        self.term_frequencies[doc_id].update(tokens)

    def get_documents(self, term: str) -> list[int]:
        
        return sorted(self.index.get(term, set()))
    def load_movies(self) ->list[Movie]:
        with open(self.DATA_PATH) as file:
            data = json.load(file)
            return data["movies"]
    def load_stopwords(self) ->list[str]:
        result: list[str] = []
        with open(self.STOPWORDS_PATH, "r", encoding="utf-8") as f:
            file_str = self.preprocess_text(f.read())
            result = file_str.splitlines()

        return result

    def preprocess_text(self, input: str) -> str:
        result = input.lower()
        
        result = result.translate(self.table)
    
        return result


    def tokenize_text(self, input: str) -> list[str]:
        tokens = self.preprocess_text(input).split()
        result: list[str] = []

        for token in tokens:
            if token and token not in self.STOPWORDS:
                result.append(self.stemmer.stem(token))
            
        return result
    def get_tf(self, doc_id: int, term: str) -> int:
        if doc_id not in self.term_frequencies:
            return 0
        return self.term_frequencies[doc_id][term]

    
    def get_idf(self, term: str) -> float:
        tok = self.tokenize_term(term)
        tf: int = len(self.index.get(tok, ()))
            
        idf = math.log( (len(self.term_frequencies) + 1) / (tf + 1))

        return idf

    def get_tfidf(self, doc_id: int, term: str) -> float:

        tf: int = self.get_tf(doc_id, term)
        idf: float = self.get_idf(term)
        return tf * idf

    def tokenize_term(self, term: str) -> str:
        token: list[str] = self.tokenize_text(term)
        if len(token) != 1:
            raise ValueError("Error tokenizing term")
        return token[0]

    def build(self) -> None:
        movies: list[Movie] = self.load_movies()

        for movie in movies:
            self.__add_document(movie["id"], f"{movie['title']} {movie["description"]}")
            self.docmap[movie["id"]] = movie

    def save(self) -> None:
        os.makedirs(self.CACHE_PATH, exist_ok=True)
        index_path: str = os.path.join(self.CACHE_PATH, "index.pkl")
        docmap_path: str = os.path.join(self.CACHE_PATH, "docmap.pkl")
        term_path: str = os.path.join(self.CACHE_PATH, "term_frequencies.pkl")
        
        with open(index_path, "wb") as f:
            pickle.dump(self.index, f)

        with open(docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)

        with open(term_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self) -> None:
        index_path: str = os.path.join(self.CACHE_PATH, "index.pkl")
        docmap_path: str = os.path.join(self.CACHE_PATH, "docmap.pkl")
        term_path: str = os.path.join(self.CACHE_PATH, "term_frequencies.pkl")

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
    i: int = 0
    query_processed: list[str] = iidx.tokenize_text(query)
    results: list[int] = []
    seen: set[int] = set()

    for qsearch in query_processed:
        response: list[int] = iidx.get_documents(qsearch)

        for doc in response:
            if i == limit:
                break
            if doc in seen:
                continue
            seen.add(doc)
            results.append(doc)
            i += 1


    for idx, key in enumerate(results):
        movie = iidx.docmap[key]
        print(f"{idx + 1}. ({movie["id"]}) {movie["title"]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build the index reverse")

    tf_parser = subparsers.add_parser("tf", help="Show the term freqency")
    tf_parser.add_argument("doc_id", type=int, help="Doc ID document for frequency")
    tf_parser.add_argument("term", type=str, help="Term frequency arg")

    idf_parser = subparsers.add_parser("idf", help="Get IDF score")
    idf_parser.add_argument("term", type=str, help="Term to get score IDF")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF score")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term for the calculation")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            search_command(args.query)
        case "build":
            i = InvertedIndex(False)
            i.build()
            i.save()
        case "tf":
            i = InvertedIndex()
            term = i.tokenize_term(args.term)
            fq = i.get_tf(args.doc_id, term)
            print(f"The frequency of {args.term} is {fq}")

        case "idf":
            i = InvertedIndex()
            idf: float = i.get_idf(args.term)
            print(f"Inverse document frequency of {args.term} : {idf:.2f}")
        case "tfidf":
            i = InvertedIndex()
            tfidf: float = i.get_tfidf(args.doc_id, args.term)
            print(f"TF-IDF score of {args.term} in document {args.doc_id} is {tfidf:.2f}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
