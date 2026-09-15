import argparse
import json
import os
import re
from pathlib import Path

from keyword_search_cli import Movie
from lib import semantic_search

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")

def load_movies() ->list[Movie]:
    with open(DATA_PATH) as file:
        data = json.load(file)
    return data["movies"]

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic search CLI")
    subparsers = parser.add_subparsers(dest="commands", help="available commands")

    verify_parser = subparsers.add_parser("verify", help="Verify model loaded")

    embed_text_parser = subparsers.add_parser("embed_text", help="Generate embedding")
    embed_text_parser.add_argument("text", type=str, help="Text used to generate emb")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verify")

    embed_query_parser = subparsers.add_parser("embed_query", help="Generate embedding query")
    embed_query_parser.add_argument("query", type=str, help="Query to embbed")

    search_parser = subparsers.add_parser("search", help="Search movies")
    search_parser.add_argument("query", type=str,help="Query used search movies")
    search_parser.add_argument("--limit", type=int, default=5, help="limit results")

    chunk_parser = subparsers.add_parser("chunk", help="Chunk texts")
    chunk_parser.add_argument("text", type=str, help="Text to divide in chunks")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Size of the chunks")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="Overlap between chunks")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Semantic chunking")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to be divided")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="Maximmun chunks")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="Overlap")

    embed_chunk_parser = subparsers.add_parser("embed_chunks", help="Semantic searching")

    search_chunked_parser = subparsers.add_parser("search_chunked", help="Semantic search chunk")

    search_chunked_parser.add_argument("query", type=str, help="Search queery")
    search_chunked_parser.add_argument("--limit", type=int, default=5, help="limit search")
    
    args = parser.parse_args()

    

    match args.commands:
        case "verify":
            semantic_search.verify_model()
        case "embed_text":
            semantic_search.embed_text(args.text)
        case "verify_embeddings":
            movies = load_movies()
            m = semantic_search.SemanticSearch()
            m.load_or_create_embeddings(movies)

            print(f"Number of docs: {len(m.documents)}")
            print(
                f"Embeddings shape: {m.embeddings.shape[0]} vectors in {m.embeddings.shape[1]} dimensions"
            )

        case "embed_query":
            semantic_search.embed_query(args.query)
        case "search":
            movies = load_movies()
            m = semantic_search.SemanticSearch()
            m.load_or_create_embeddings(movies)

            result = m.search(args.query, args.limit)

            i = 1

            for r in result:
                print(f"{i}. {r[1]['title']} (score: {r[0]})\n {r[1]['description']}")
                i += 1

        case "chunk":
            splitted_text: list[str] = args.text.split(" ")

            chunks_size: int = args.chunk_size
            overlap: int = args.overlap
            offset: int = 0
            i: int = 1

            print(f"Chunking {len(args.text)} characters")

            while(offset < len(splitted_text)):
                r = " "
                if overlap > 0:
                    if i == 1:
                        r = r.join(splitted_text[offset:chunks_size + offset])
                    else:
                        r = r.join(splitted_text[offset-overlap: chunks_size + offset])
                else:
                    r = r.join(splitted_text[offset:chunks_size + offset])
                print(f"{i}. {r}")
                offset += chunks_size
                i += 1

        case "semantic_chunk":
            chunks: list[str] = re.split(r"(?<=[.!?])\s+", args.text)

            max_size = args.max_chunk_size
            overlap = args.overlap

            print(f"Semantically chunking {len(args.text)} characters")

            result: list[str] = []
            offset: int = 0


            while (offset < len(chunks)):
                chunk_sentences = chunks[offset: offset + max_size]

                if result and len(chunk_sentences) <= overlap:
                    break
                result.append(" ".join(chunk_sentences))
                offset += max_size - overlap

            i: int = 1
            for s in result:
                print(f"{i}. {s}")
                i += 1

        case "embed_chunks":
            movies = load_movies()
            
            sm = semantic_search.ChunkedSemanticSearch()
            embeddings = sm.load_or_create_chunk_embeddings(movies)

            print(f"Generated {len(embeddings)} chunked embeddings")

        case "search_chunked":
            movies = load_movies()
            sm = semantic_search.ChunkedSemanticSearch()

            emb = sm.load_or_create_chunk_embeddings(movies)

            r = sm.search_chunks(args.query, args.limit)

            for idx, doc in enumerate(r):
                print(f"{idx + 1}. {doc["title"]} (score: {doc["score"]:.4f})")
                print(f"  {doc["document"]}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

