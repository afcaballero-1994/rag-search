import os
from sentence_transformers import SentenceTransformer
import numpy as np
from operator  import itemgetter
import numpy.typing as npt
from numpy.typing import NDArray
from typing import Any
from pathlib import Path

SCORE_PRECISION = 3

import re
import json

EmbeddingArray = NDArray[Any]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
CACHE_PATH = os.path.join(PROJECT_ROOT, "cache")

class SemanticSearch:
    file_path = os.path.join(CACHE_PATH, "movie_embeddings.npy")

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents: list | None = None
        self.document_map = {}

    def generate_embedding(self, text: str):
        if not text:
            raise ValueError("provide no empty String")
        
        return self.model.encode([text])[0]

    def build_embeddings(self, documents: list[dict]) -> npt.ArrayLike:
        self.documents = documents
        rmovie: list[str] = []

        for document in documents:
            self.document_map[document['id']] = document
            rmovie.append(f"{document['title']}: {document['description']}")

        self.embeddings = self.model.encode(rmovie, show_progress_bar=True)

        with open(self.file_path, 'wb') as f:
            np.save(f, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]):
        self.documents = documents

        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(self.file_path):
            self.embeddings = np.load(self.file_path)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        
        return self.build_embeddings(documents)

    def search(self, query: str, limit: int = 5):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call load_or_create_embeddings first")
        q_embeddings = self.generate_embedding(query)

        result = []

        for i, doc_embedding in enumerate(self.embeddings):
            score = cosine_similarity(q_embeddings, doc_embedding)
            r = (score, self.documents[i])
            result.append(r)

        result.sort(key=itemgetter(0), reverse=True)

        return result[0:limit]

def verify_model() -> None:
    m = SemanticSearch()

    print(f"Model loaded: {m.model} Max sequence length: {m.model.max_seq_length}")
        
def embed_text(text: str):
    m = SemanticSearch()

    r = m.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {r[:3]}")
    print(f"Dimensions: {r.shape[0]}")


def embed_query(query: str):
    m = SemanticSearch()
    r = m.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 3 dimensions: {r[:3]}")
    print(f"Shape: {r.shape}")

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    d_product = np.dot(vec1, vec2)
    n1 = np.linalg.norm(vec1)
    n2 = np.linalg.norm(vec2)

    if n1 == 0 or n2 == 0:
        return 0.0
    return d_product / (n1 * n2)

def semantic_chunking(text: str, max_chunk_size: int = 4, overlap:int = 0) -> list[str]:
    text = text.strip()

    if len(text) == 0:
        return []
    
    chunks: list[str] = re.split(r"(?<=[.!?])\s+", text)

    if len(chunks) == 1 and not text.endswith((".", "!", "?")):
        chunks = [text]

    max_size = max_chunk_size
    overlap = overlap

    print(f"Semantically chunking {len(text)} characters")

    result: list[str] = []
    offset: int = 0


    while (offset < len(chunks)):
        chunk_sentences = chunks[offset: offset + max_size]

        if result and len(chunk_sentences) <= overlap:
            break


        cleaned_sentences = []
        for chunk_sentence in chunk_sentences:
            chunk_sentence = chunk_sentence.strip()
            if chunk_sentence:
                cleaned_sentences.append(chunk_sentence)
        if not cleaned_sentences:
            offset += max_size - overlap
            continue
        
        result.append(" ".join(cleaned_sentences))
        offset += max_size - overlap

    return result

class ChunkedSemanticSearch(SemanticSearch):
    file_path = os.path.join(CACHE_PATH, "chunk_embeddings.npy")
    json_path = os.path.join(CACHE_PATH, "chunk_metadata.json")
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        chunks: list[str] = []
        meta: list[dict] = []

        for idx, document in enumerate(documents):
            self.document_map[document['id']] = document
            if not document['description']:
                continue
            tmp = semantic_chunking(document['description'], overlap=1)
            for i, chk in enumerate(tmp):
                chunks.append(chk)
                mtmp = {}
                mtmp["movie_idx"] = idx
                mtmp["chunk_idx"] = i
                mtmp["total_chunks"] = len(tmp)
                meta.append(mtmp)
                
        

        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)
        self.chunk_metadata = meta

        with open(self.file_path, 'wb') as f:
            np.save(f, self.chunk_embeddings)

        with open(self.json_path, 'w') as f:
            json.dump({"chunks": meta, "total_chunks": len(chunks)}, f, indent=2)
            
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents

        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(self.json_path):
            with open(self.json_path, "r") as f:
                data = json.load(f)
                self.chunk_metadata = data["chunks"]

        if os.path.exists(self.file_path):
            self.chunk_embeddings = np.load(self.file_path)
            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        q_embbedings = self.generate_embedding(query)

        chunk_scores = []

        for idx, chk in enumerate(self.chunk_embeddings):
            score = cosine_similarity(chk, q_embbedings)
            tmp_dict = {
                "chunk_idx": self.chunk_metadata[idx]["chunk_idx"],
                "movie_idx": self.chunk_metadata[idx]["movie_idx"],
                "score": score
            }

            chunk_scores.append(tmp_dict)

        result_scores = {}

        for chk in chunk_scores:
            if not chk["movie_idx"] in result_scores:
                result_scores[chk["movie_idx"]] = chk["score"]
            else:
                if chk["score"] > result_scores[chk["movie_idx"]]:
                    result_scores[chk["movie_idx"]] = chk["score"]

        sresults = dict(sorted(result_scores.items(), key=itemgetter(1), reverse=True)[:limit])

        result = []

        for k,v in sresults.items():
            result.append(
                {
                    "id": self.documents[k]["id"],
                    "title": self.documents[k]["title"],
                    "document": self.documents[k]["description"][:100],
                    "score": round(v, SCORE_PRECISION),
                    "metadata": self.documents[k].get("metadata", {})
                }
            )

        return result
        
