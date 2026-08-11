from sentence_transformers import SentenceTransformer

class SemanticSearch:

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)

    def generate_embedding(self, text: str):
        if not text:
            raise ValueError("provide no empty String")
        
        embeddings = self.model.encode([text])

        print(type(embeddings))
        print(type(embeddings[0]))

        return embeddings[0]
        

def verify_model() -> None:
    m = SemanticSearch()

    print(f"Model loaded: {m.model} Max sequence length: {m.model.max_seq_length}")
        
def embed_text(text: str):
    m = SemanticSearch()

    r = m.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {r[:3]}")
    print(f"Dimensions: {r.shape[0]}")
