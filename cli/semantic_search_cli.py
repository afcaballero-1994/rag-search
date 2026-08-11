import argparse

from lib import semantic_search

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic search CLI")
    subparsers = parser.add_subparsers(dest="commands", help="available commands")

    verify_parser = subparsers.add_parser("verify", help="Verify model loaded")

    embed_text_parser = subparsers.add_parser("embed_text", help="Generate embedding")
    embed_text_parser.add_argument("text", type=str, help="Text used to generate emb")
    
    args = parser.parse_args()

    

    match args.commands:
        case "verify":
            semantic_search.verify_model()
        case "embed_text":
            semantic_search.embed_text(args.text)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
