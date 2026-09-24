from PIL import Image
from typing import Any
from numpy._typing import NDArray
from sentence_transformers import SentenceTransformer
from lib.semantic_search import cosine_similarity
import os


def verify_image_embedding(image_path: str):
    mmsearch = MultimodalSearch()
    embedding = mmsearch.embed_image(image_path)

    print(f"Embedding shape: {embedding.shape[0]} dimensions")

class MultimodalSearch:
    def __init__(self, documents: list[dict] | None = None ,model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name, device="cuda")
        self.documents = documents
        self.texts = []
        if self.documents:
            for doc in self.documents:
                self.texts.append(f"{doc['title']}: {doc['description']}")

        self.texts_embeddings = self.model.encode(self.texts, show_progress_bar=True)
        

    def embed_image(self, image_path: str) -> NDArray[Any]:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        img = Image.open(image_path)

        img_embeddings = self.model.encode(img)

        return img_embeddings

    def search_with_image(self, image_path: str):
        img_embeddings = self.embed_image(image_path)
        results = []
        if self.documents is None:
            raise ValueError("Not documents provided to Multimodal Search constructor")

        for idx,  texts_embedding in enumerate(self.texts_embeddings):
            score = cosine_similarity(img_embeddings, texts_embedding)
            results.append({
                "id": self.documents[idx]["id"],
                "title": self.documents[idx]["title"],
                "document": self.documents[idx]["description"],
                "score": score,
            })

        return sorted(results, key=lambda x: x["score"], reverse=True)[:5]
