import os
from dotenv import load_dotenv
from openai import OpenAI

def main() -> None:
   load_dotenv()

   api_key = os.environ.get("OPENROUTER_API_KEY")
   if not api_key:
       raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

   client = OpenAI(
       base_url="https://openrouter.ai/api/v1",
       api_key=api_key,
   )

   response = client.chat.completions.create(
       model="openrouter/free",
       messages = [
           {
               "role": "user",
               "content": "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum."
           }
       ]
   )

   if response.usage is None:
      raise RuntimeError("API response no usage data")

   r_string = response.choices[0].message.content

   print(f"Prompt tokens: {response.usage.prompt_tokens}\nResponse tokens: {response.usage.completion_tokens}")
   print(r_string)

if __name__ == "__main__":
    main()
 
