import os
from operator import itemgetter
from itertools import zip_longest

from torch import chunk
from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch

def hybrid_score(bm25_score: float, semantic_score: float, alpha=0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score

def normalize_command(scores: list[float]) -> list[float]:
    if not scores:
        return []
    min_score: float = min(scores)
    max_score: float = max(scores)

    if min_score == max_score:
        return [1.0]
    result: list[float] = []

    for score in scores:
        result.append( (score - min_score) / (max_score - min_score))

    return result

class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()

        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()


    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        bmresults = self._bm25_search(query, limit)
        chunk_results = self.semantic_search.search_chunks(query, limit)


        bm25_scores = list(map(itemgetter("score"), bmresults))
        chunk_scores = list(map(itemgetter("score"), chunk_results))

        bm25_normalized = normalize_command(bm25_scores)
        chunk_normalized = normalize_command(chunk_scores)

        results: dict[int, tuple] = {}

        for idx, bm25_result in enumerate(bmresults):
            doc_id = bm25_result["doc_id"]
            if doc_id not in results:
                results[doc_id] = (bm25_normalized[idx], 0.0)
            else:
                results[doc_id] = (bm25_normalized[idx], results[doc_id][1])

        for idx, chunk_result in enumerate(chunk_results):
            doc_id = chunk_result["id"]
            if doc_id not in results:
                results[doc_id] = (0.0, chunk_normalized[idx])
            else:
                results[doc_id] = (results[doc_id][0], chunk_normalized[idx])

        for k, v in results.items():
            hy_score = hybrid_score(v[0], v[1])

            results[k] = (v[0], v[1], hy_score)

        sorted_data = sorted(results.items(), key=lambda kv: kv[1][2], reverse=True)[0:limit]

        response = []
        for d in sorted_data:
            doc_id = d[0]
            bm25_score = d[1][0]
            semantic_score = d[1][1]
            hybrid_scor = d[1][2]

            response.append(
                {
                    "id": doc_id,
                    "title": self.semantic_search.document_map[doc_id]["title"],
                    "bm25_score": bm25_score,
                    "semantic_score": semantic_score,
                    "hybrid_score": hybrid_scor,
                    "description": self.semantic_search.document_map[doc_id]["description"][:100]
                }
            )

        return response
            
            

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        raise NotImplementedError("RRF hybrid search not implemented yet.")
