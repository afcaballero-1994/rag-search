import argparse
import  base64
import mimetypes

from hybrid_search_cli import load_model, MODEL


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal search CLI")

    parser.add_argument("--image", type=str, help="Path image to submit")
    parser.add_argument("--query", type=str, help="Query to be used for the search")

    args = parser.parse_args()

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    with open(args.image, "rb") as f:
        data = f.read()

    client = load_model()
    system_prompt = f"""Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary"""
    data_url = f"data:{mime};base64,{base64.b64encode(data).decode()}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": system_prompt.strip()},
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": args.query.strip()},
            ]
        }
    ]
    response = client.chat.completions.create(model=MODEL, messages=messages)

    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("Error getting a response from openrouter")
    print(f"Rewritten query: {content.strip()}")
    if response.usage is not None:
        print(f"Total tokens:   {response.usage.total_tokens}")
    

if __name__ == "__main__":
    main()
