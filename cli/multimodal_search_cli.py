from lib.multimodal_search import MultimodalSearch, verify_image_embedding
from hybrid_search_cli import load_movies
import argparse

def main():
    parser = argparse.ArgumentParser(description="Multimodal mode CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_embedding_parser = subparsers.add_parser(
        "verify_image_embedding", help="Verify that image was embedded correctly"
    )
    verify_image_embedding_parser.add_argument("image_path", type=str, help="Path to where the image is located")

    image_search_parser = subparsers.add_parser("image_search", help="Make search using an image")
    image_search_parser.add_argument("image", type=str, help="Path to image location")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image_path)
        case "image_search":
            movies = load_movies()
            mmsearch = MultimodalSearch(movies)
            response = mmsearch.search_with_image(args.image)

            for idx, doc in enumerate(response, 1):
                print(f"{idx}. {doc['title']} (similarity: {doc['score']:.4f})\n {doc['document'][:100]}...")
        case _:
            raise RuntimeError(f"{args.command} not implemented")

if __name__ == "__main__":
    main()
