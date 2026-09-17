import argparse

from lib.keyword_search import *

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build the index reverse")

    tf_parser = subparsers.add_parser("tf", help="Show the term freqency")
    tf_parser.add_argument("doc_id", type=int, help="Doc ID document for frequency")
    tf_parser.add_argument("term", type=str, help="Term frequency arg")

    idf_parser = subparsers.add_parser("idf", help="Get IDF score")
    idf_parser.add_argument("term", type=str, help="Term to get score IDF")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF score")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term for the calculation")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25")

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score given document id and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?',
                                default=BM25_K1, help="Tunable k1 BM25 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable b parameter BM25")

    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full bm25 scoring"
    )

    bm25search_parser.add_argument(
        "query", type=str, help="Search query"
    )
    bm25search_parser.add_argument(
        "limit", type=int, nargs='?', default=5, help="Limit for the results list"
    )

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            search_command(args.query)
        case "build":
            i = InvertedIndex(False)
            i.build()
            i.save()
        case "tf":
            i = InvertedIndex()
            term = i.tokenize_term(args.term)
            fq = i.get_tf(args.doc_id, term)
            print(f"The frequency of {args.term} is {fq}")

        case "idf":
            i = InvertedIndex()
            idf: float = i.get_idf(args.term)
            print(f"Inverse document frequency of {args.term} : {idf:.2f}")
        case "tfidf":
            i = InvertedIndex()
            tfidf: float = i.get_tfidf(args.doc_id, args.term)
            print(f"TF-IDF score of {args.term} in document {args.doc_id} is {tfidf:.2f}")
        case "bm25idf":
            i = InvertedIndex()
            term = i.tokenize_term(args.term)
            score = i.get_bm25_idf(term)
            print(f"BM25 IDF score of '{args.term}': {score:.2f}")
        case "bm25tf":
            i = InvertedIndex()
            term = i.tokenize_term(args.term)
            doc_id = args.doc_id
            k1 = args.k1
            B = args.b

            bm25tf = i.get_bm25_tf(doc_id, term, k1, B)

            print(f"BM25 TF score of '{term}' in document '{doc_id}':{bm25tf:.2f}")
        case "bm25search":
            i = InvertedIndex()
            result = i.bm25_search(args.query, args.limit)

            for i, res in enumerate(result, 1):
                print(f"{i}. ({res['doc_id']}) {res['title']} - Score: {res['score']:.2f}")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
