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
from operator import itemgetter

import pickle
import sys
import math

BM25_K1: float = 1.5
BM25_B: float = 0.75

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
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
    CACHE_PATH = os.path.join(PROJECT_ROOT, "cache")
    STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
    doc_lengths_path = os.path.join(CACHE_PATH, "doc_lengths.pkl")
    index_path: str = os.path.join(CACHE_PATH, "index.pkl")
    STOPWORDS: set[str]

    def __init__(self, load_cache: bool = True):
        self.STOPWORDS = set(self.preprocess_text(x) for x in self.load_stopwords())
        self.stemmer = PorterStemmer()
        self.doc_lengths: dict[int, int] = {}
        if load_cache:
            self.load()


    def __add_document(self, doc_id: int, text: str) -> None:
        tokens: list[str] = self.tokenize_text(text)

        self.doc_lengths[doc_id] = len(tokens)

        for token in set(tokens):
            self.index[token].add(doc_id)

        self.term_frequencies[doc_id].update(tokens)

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        avg: float = 0.0
        total_docs: int = len(self.doc_lengths)
        total_tok: int = 0

        for num_toks in self.doc_lengths.values():
            total_tok += num_toks

        avg = total_tok / total_docs

        return avg

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
        text = self.preprocess_text(input)
        tokens = text.split()
        valid_tokens: list[str] = []

        for token in tokens:
            if token:
                valid_tokens.append(token)
        filtered_words = []

        for word in valid_tokens:
            if word not in self.STOPWORDS:
                filtered_words.append(word)
        stemmer = PorterStemmer()
        stemmed_words = []

        for word in filtered_words:
            stemmed_words.append(stemmer.stem(word))
            
        return stemmed_words
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

    def get_bm25_idf(self, term: str) -> float:
        doc_count: int = len(self.docmap)
        term_doc_count: int = len(self.index.get(term, ()))

        bm25: float = math.log( (doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1)
        return bm25

    def get_bm25_tf(self, doc_id: int, term: str, k1:float=BM25_K1, b:float=BM25_B) -> float:
        tf = self.get_tf(doc_id, term)
        length_norm: float = 1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())
        

        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def bm25(self, doc_id: int, term: str) -> float:
        idf: float = self.get_bm25_idf(term)
        tf: float = self.get_bm25_tf(doc_id, term)

        return idf * tf

    def bm25_search(self, query: str, limit: int) -> list[dict]:
        result: dict[int, float] = {}
        query_tokens: list[str] = self.tokenize_text(query)

        for query in query_tokens:

            docs_id = self.get_documents(query)

            for doc in docs_id:
                if doc not in result:
                    result[doc] = self.bm25(doc, query)
                else:
                    result[doc] += self.bm25(doc, query)

        sorted_data = sorted(result.items(), key=itemgetter(1), reverse=True)[0:limit]

        results = []
        for doc_id, score in sorted_data:
            doc = self.docmap[doc_id]
            results.append(
                {
                    "doc_id": doc["id"],
                    "title": doc["title"],
                    "document": doc["description"][0:100],
                    "score": score
                }
            )
            

        return results

    def tokenize_term(self, term: str) -> str:
        token: list[str] = self.tokenize_text(term)
        if len(token) != 1:
            print(term)
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

        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

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

        try:
            with open(self.doc_lengths_path, "rb") as f:
                self.doc_lengths = pickle.load(f)
        except FileNotFoundError:
            print(f"Error: file {self.doc_lengths_path} does not exist")
            
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
