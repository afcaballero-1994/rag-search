import argparse
import logging

from evaluation_cli import load_movies
from hybrid_search_cli import HybridSearch, load_model, MODEL

def get_results(query: str, limit: int = 5) -> list[dict]:
    movies = load_movies()
    mhs = HybridSearch(movies)
    return mhs.rrf_search(query, 60, limit * 5)[:limit]

def main() -> None:
    logging.basicConfig(filename="rff_search.log", level=logging.INFO)
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Provide a summary got from the results"
    )
    summarize_parser.add_argument("query", type=str, help="Search query to get summaries from")
    summarize_parser.add_argument("--limit", type=int, nargs="?", default=5, help="Maximmun results")

    citations_parser = subparsers.add_parser(
        "citations", help="Get a summary including citations"
    )
    citations_parser.add_argument("query", type=str, help="Search query to get summary with citation")
    citations_parser.add_argument("--limit", type=int, nargs="?", default=5, help="Maximmun number results default=5")

    questions_parser = subparsers.add_parser("question", help="Perform a search and try to find an answer")
    questions_parser.add_argument("question", type=str, help="Question you may have")
    questions_parser.add_argument("--limit", type=int, nargs="?", default=5, help="Maximmun number results default=5")

    args = parser.parse_args()
    client = load_model()
    match args.command:
        case "rag":
            query = args.query
            results_search = get_results(query)
            prompt = f"""You are a RAG agent for Webflyx, a movie streaming service.
Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
Provide a comprehensive answer that addresses the user's query.

Query: {query}

Documents:
{results_search}

Answer:"""
            response = client.chat.completions.create(
                model=MODEL, messages=[{"role": "user", "content": prompt}]
            )

            message = response.choices[0].message.content
            if message is None:
                raise RuntimeError("No valid response")

            print("Search Results:")
            for doc in results_search:
                print(f"- {doc["title"]}")
            print(f"RAG Response:\n{message.strip()}")
        case "summarize":
            query = args.query
            limit = args.limit
            results = get_results(query, limit)
            prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

This should be tailored to Webflyx users. Webflyx is a movie streaming service.

Query: {query}

Search results:
{results}

Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""
            response = client.chat.completions.create(
                model=MODEL, messages=[{"role": "user", "content": prompt}]
            )

            message = response.choices[0].message.content
            if message is None:
                raise RuntimeError("No valid response")
            print("Search Results:")
            for doc in results:
                print(f"- {doc["title"]}")
            print(f"LLM Summary:\n{message.strip()}")
        case "citations":
            query = args.query
            limit = args.limit
            results = get_results(query, limit)
            prompt = f"""Answer the query below and give information based on the provided documents.

The answer should be tailored to users of Webflyx, a movie streaming service.
If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

Query: {query}

Documents:
{results}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources in the format [1], [2], etc. when referencing information
- On your answer, please include the title or the source used to get that information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the provided documents, say "I don't have enough information"
- Be direct and informative

Answer:"""
            response = client.chat.completions.create(
                model=MODEL, messages=[{"role": "user", "content": prompt}]
            )

            message = response.choices[0].message.content
            if message is None:
                raise RuntimeError("No valid response")
            print("Search Results:")
            for doc in results:
                print(f"- {doc["title"]}")
            print(f"LLM Answer:\n{message.strip()}")
        case "question":
            question = args.question
            limit = args.limit
            results = get_results(question, limit)
            context = ""
            for idx, doc in enumerate(results, 1):
                context += f"{idx}. {doc["title"]}: {doc["document"]}\n\n"
            
            prompt = f"""Answer the user's question based on the provided movies that are available on Webflyx, a streaming service.

Question: {question}

Documents:
{context}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""
            
            response = client.chat.completions.create(
                model=MODEL, messages=[{"role": "user", "content": prompt}]
            )

            message = response.choices[0].message.content
            if message is None:
                raise RuntimeError("No valid response")
            print("Search Results:")
            for doc in results:
                print(f"- {doc["title"]}")
            print(f"Answer:\n{message.strip()}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
