import ollama

# Step 1 — Retrieve all stored chunks from Chroma
all_docs = collection.get()['documents']  # This is a list of all chunk texts

# Step 2 — Join them into a single context string
context = "\n\n".join(all_docs)

# Step 3 — Ask Ollama
prompt = f"""
You are given the following resume text:

{context}

Extract only the programming languages mentioned.
Return them as a clean JSON list in the format:
{{
  "programming_languages": ["language1", "language2", ...]
}}
"""

response = ollama.chat(
    model="llama3",  # or any other model you have in Ollama
    messages=[
        {"role": "system", "content": "You are an information extraction assistant."},
        {"role": "user", "content": prompt}
    ]
)

print(response['message']['content'])
