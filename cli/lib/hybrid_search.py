import os
from operator import itemgetter

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch

def hybrid_score(bm25_score: float, semantic_score: float, alpha=0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score

def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)

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

    def weighted_search(self, query: str, alpha: float = 0.5, limit: int = 5) -> list[dict]:
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
            hy_score = hybrid_score(v[0], v[1], alpha)

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
        bm25_results = self._bm25_search(query, limit)
        semantic_results = self.semantic_search.search_chunks(query, limit)

        combined_results: dict = {}

        for idx, bm25_result in enumerate(bm25_results, 1):
            doc_id = bm25_result["doc_id"]

            if doc_id not in combined_results:
                combined_results[doc_id] = {
                    "title": bm25_result["title"],
                    "description": bm25_result["document"],
                    "bm25_rank": idx,
                    "semantic_rank": 0,
                    "rrf_score": rrf_score(idx, k)
                }
            else:
                combined_results[doc_id]["bm25_rank"] = idx
                combined_results[doc_id]["rrf_score"] = combined_results[doc_id]["rrf_score"] + rrf_score(idx, k)

        for idx, semantic_result in enumerate(semantic_results, 1):
            doc_id = semantic_result["id"]

            if doc_id not in combined_results:
                combined_results[doc_id] = {
                    "title": semantic_result["title"],
                    "description": semantic_result["document"],
                    "semantic_rank": idx,
                    "rrf_score": rrf_score(idx, k),
                    "bm25_rank": 0
                }
            else:
                combined_results[doc_id]["semantic_rank"] = idx
                combined_results[doc_id]["rrf_score"] = combined_results[doc_id]["rrf_score"] + rrf_score(idx, k)


        response = []
        for doc_id, data in combined_results.items():
            tmp = {
                "id": doc_id,
                "title": data["title"],
                "document": data["description"],
                "semantic_rank": data["semantic_rank"],
                "bm25_rank": data["bm25_rank"],
                "rrf_score": data["rrf_score"]
            }
            response.append(tmp)

        return sorted(response, key=lambda x: x["rrf_score"], reverse=True)
