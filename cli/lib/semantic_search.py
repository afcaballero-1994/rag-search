import os
from sentence_transformers import SentenceTransformer
import numpy as np
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
        self.documents = None
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

def verify_model() -> None:
    m = SemanticSearch()

    print(f"Model loaded: {m.model} Max sequence length: {m.model.max_seq_length}")
        
def embed_text(text: str):
    m = SemanticSearch()

    r = m.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {r[:3]}")
    print(f"Dimensions: {r.shape[0]}")
