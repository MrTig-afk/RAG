import warnings
import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter
import ollama
from pyserini.search.lucene import LuceneSearcher

warnings.filterwarnings("ignore", category=FutureWarning)

# --- 1. Read PDF and convert to Markdown ---
md_text = pymupdf4llm.to_markdown(
    r"D:\MSDS 2nd Sem\Case Studies in Data Science\Assignments\WIL\Resume for Thomas Davis.pdf"
)

# --- 2. Split text into chunks ---
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""]
)
chunks = splitter.split_text(md_text)

# --- 3. Instead of building index, use prebuilt LuceneSearcher ---
searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')

# --- 4. Search using your query ---
query = "List all the companies the person has previously worked in."
hits = searcher.search(query, k=5)

# --- 5. Retrieve the top hit documents' contents ---
retrieved_chunks = []
for hit in hits:
    doc = searcher.doc(hit.docid)
    retrieved_chunks.append(doc.raw())

retrieved_text = "\n\n".join(retrieved_chunks)

# --- 6. Send retrieved text to Ollama for extraction ---
ollama_prompt = f"""
You are given the following resume excerpts:

{retrieved_text}

Extract the companies the person has previously worked in.
Return them as a JSON list in the format:
{{"companies": ["Company1", "Company2", ...]}}
"""

response = ollama.chat(
    model="llama3.2:3b",
    messages=[
        {"role": "system", "content": "You are an expert at extracting structured information from text."},
        {"role": "user", "content": ollama_prompt}
    ]
)

print("\nOllama Output:")
print(response["message"]["content"])
