import warnings
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
import pymupdf4llm
import chromadb
import ollama

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# 1. Load local embedding model
model = SentenceTransformer("./all-MiniLM-L6-v2")

# 2. Read PDF and convert to Markdown
md_text = pymupdf4llm.to_markdown(
    r"D:\MSDS 2nd Sem\Case Studies in Data Science\Assignments\WIL\Lecun98.pdf"
)
print(md_text)

# 3. Split text into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)
chunks = splitter.split_text(md_text)

# 4. Generate embeddings
embeddings = model.encode(chunks, show_progress_bar=True)
print(f"Embeddings shape: {embeddings.shape}")

# 5. Create Chroma persistent DB
chroma_client = chromadb.PersistentClient(path="./chroma_resume_db")

# Clear existing collection
if "resume_embeddings" in [c.name for c in chroma_client.list_collections()]:
    chroma_client.delete_collection("resume_embeddings")

collection = chroma_client.create_collection(name="resume_embeddings")

# 6. Store chunks in Chroma
collection.add(
    documents=chunks,
    embeddings=embeddings.tolist(),
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    metadatas=[{"source": "resume"} for _ in chunks]
)

print(f"Stored {collection.count()} chunks")

# 7. Vector search in Chroma
query = "Extract the keywords from the paper"
query_embedding = model.encode([query]).tolist()[0]

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5  # Increase if you want more context
)

retrieved_text = "\n\n".join(results["documents"][0])

# 8. Send retrieved text to Ollama for structured extraction
ollama_prompt = f"""
You are given the following excerpt:

{retrieved_text}
Extract the keywords from the same, DO NOT add or remove any information of your own. If you are unsure, simply return I do not know
"""

response = ollama.chat(
    model="llama3.2:3b",  # Change this to your Ollama model
    messages=[
        {"role": "system", "content": "You are an expert at extracting structured information from text."},
        {"role": "user", "content": ollama_prompt}
    ]
)

print("\nOllama Output:")
print(response["message"]["content"])
