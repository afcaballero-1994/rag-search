import argparse
import json

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            with open("data/movies.json") as file:
                data = json.load(file)
                i: int = 0
                limit: int = 5
                for v in data["movies"]:
                    if args.query in v["title"]: 
                        print(f"{i + 1}. {v["title"]}")
                        i += 1
                        if i == limit:
                            break

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
