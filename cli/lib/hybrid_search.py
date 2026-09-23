import json
import os, logging
from operator import itemgetter

logger = logging.getLogger(__name__)

from typing import Literal

from openai import OpenAI
from dotenv import load_dotenv

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch

from sentence_transformers import CrossEncoder


MODEL = "dots-studio/dots-3-note-preview:free"


def get_prompt(query: str, method: str) -> str | None:
    ENHANCED_METHODS = {
    "spell": f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{query}"
""",
    "rewrite": f"""Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{query}"
""",
"expand": f"""Expand the user-provided movie search query below with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Output only the additional terms; they will be appended to the original query.

Examples:
- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

User query: "{query}"
"""
}
    if method not in ENHANCED_METHODS:
        return None
    return ENHANCED_METHODS[method]

def get_prompt_rerank(query: str, doc: dict = dict(),
                      method: Literal["individual", "batch"] | None = None,
                      doc_list_str: list | str | None = None) -> str | None:
    if method is None:
        return None
    RERANK_METHOD = {
        "individual": f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("document", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.

Score:""",
        "batch": f"""Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:
{doc_list_str}

Return the movie IDs in order of relevance, best match first.

Your response must be a raw JSON array of integers.
Do not wrap the JSON in Markdown. Do not use a ```json code block.
Do not include any explanatory text.

For example:
[75, 12, 34, 2, 1]

Ranking:"""
    }
    if method not in RERANK_METHOD:
        return None
    return RERANK_METHOD[method]

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

def enhance_query(query: str, method: Literal["spell", "rewrite", "expand"] | None = None) -> str:
    if method is None:
        return query
    prompt = get_prompt(query, method)
    if prompt is None:
        return query
    
    load_dotenv()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
       raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    client = OpenAI(
       base_url="https://openrouter.ai/api/v1",
       api_key=api_key,
   )

    response = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}]
    )

    enhanced_query: str | None = response.choices[0].message.content

    if enhanced_query is not None:
        print(f"Enhanced query ({method}): '{query}' -> '{enhanced_query}'\n")
        return enhanced_query

    return query

def rerank_results_individual(query: str,
                   results: list[dict],
                   client: OpenAI,
                   ) -> list[dict]:
    result = []
    
    for doc in results:
        prompt = get_prompt_rerank(query, doc, method="individual")
        if prompt is None:
            raise ValueError("Not able to get a prompt for individual method")

        response = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}]
        )
        data: str | None  = response.choices[0].message.content
        if data is None:
            raise ValueError("Score was not valid for rerank individual")
        print(data)
        doc["rerank_score"] = int(data.strip())
        result.append(doc)

    return result


def rerank_results_batch(query: str,
                   results: list[dict],
                   client: OpenAI,
                   ) -> list[dict]:
    doc_map: dict = {}
    doc_list: list[str] = []

    for doc in results:
        doc_id = doc["id"]
        doc_map[doc_id] = doc
        doc_list.append(
            f"{doc_id}: {doc.get("title", "")} - {doc.get("document", "")[:200]}"
        )
    docs_str = "\n".join(doc_list)
    prompt = get_prompt_rerank(query, method="batch", doc_list_str=docs_str)
    if prompt is None:
        raise RuntimeError("Not able to get prompt")

    response = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}]
    )
    data: str | None  = response.choices[0].message.content
    if data is None:
        raise RuntimeError("No response received from model")
    scores = json.loads(data)

    reranked = []
    for idx, d_id in enumerate(scores):
        if d_id in doc_map:
            reranked.append(
                {
                    **doc_map[d_id], "rerank_score": idx + 1
                }
            )
    return reranked

def rerank_cross_encoder(query: str, response: list[dict]) -> list[dict]:
    pairs = []
    for doc in response:
        pairs.append([query, f"{doc.get("title", "")} - {doc.get("document", "")}"])

    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")

    scores = cross_encoder.predict(pairs)
    print(len(scores))

    result = []

    for idx, doc in enumerate(response):
        result.append(
            {
                **doc, "cross_encoder_score": scores[idx]
            }
        )

    return result

def load_model() -> OpenAI:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
       raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    return client

def rerank(query: str, results: list[dict], method: Literal["individual", "batch", "cross_encoder"]) -> list[dict]:
    client = load_model()
    
    match method:
        case "individual":
            re = rerank_results_individual(query, results, client)
            return sorted(re, key=lambda x: x["rerank_score"], reverse=True)
        case "batch":
            re = rerank_results_batch(query, results, client)
            return sorted(re, key=lambda x: x["rerank_score"], reverse=False)
        case "cross_encoder":
            re = rerank_cross_encoder(query, results)
            return sorted(re, key=lambda x: x["cross_encoder_score"], reverse=True)
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
            
            

    def rrf_search(self, query: str, k: int, limit: int = 10,
                   method: Literal["spell", "rewrite", "expand"] | None = None,
                   rerank_method: Literal["individual", "batch", "cross_encoder"] | None = None
                   ) -> list[dict]:
        logger.info(f"Original Query: {query}")
        query = enhance_query(query, method)
        logger.info(f"Enhanced query: {query}")
        bm25_results = self._bm25_search(query, limit * 5)
        semantic_results = self.semantic_search.search_chunks(query, limit * 5)

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

        logger.info(f"RRF results: {response}\n")

        if rerank_method is not None:
            r = rerank(query, response, rerank_method)
            logger.info(f"Reranked results with {rerank_method}: {r}\n")
            return r

        sorted_results = sorted(response, key=lambda x: x["rrf_score"], reverse=True)[:limit]
        logger.info(f"Results ordered: {sorted_results}\n")
        return sorted_results
