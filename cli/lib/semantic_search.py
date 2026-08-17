import os
from sentence_transformers import SentenceTransformer
import numpy as np
from operator  import itemgetter
import numpy.typing as npt
from numpy.typing import NDArray
from typing import Any
from pathlib import Path

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
            for k, _ in doc.items():
                self.document_map[k] = doc

        if os.path.exists(self.file_path):
            self.embeddings = np.load(self.file_path)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        else:
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
