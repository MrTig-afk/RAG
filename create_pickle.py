import os
import pickle
import pdfplumber
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

PDF_FOLDER = "pdfs"  # folder with your PDFs
VECTORSTORE_PATH = "vectorstore.pkl"

# Extract text from PDFs using pdfplumber
def get_pdf_text_with_metadata(pdf_paths):
    texts = []
    metadatas = []
    for pdf_path in pdf_paths:
        pdf_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    pdf_text += page_text + "\n"
        if pdf_text.strip():  # only add if text exists
            texts.append(pdf_text)
            metadatas.append({"source": os.path.basename(pdf_path)})
    return texts, metadatas

# Split text into chunks while preserving metadata
def get_text_chunks_with_metadata(texts, metadatas, chunk_size=1000, chunk_overlap=200):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    all_chunks = []
    all_chunk_metadata = []
    for text, metadata in zip(texts, metadatas):
        chunks = splitter.split_text(text)
        all_chunks.extend(chunks)
        all_chunk_metadata.extend([metadata]*len(chunks))
    return all_chunks, all_chunk_metadata

def main():
    pdf_paths = [os.path.join(PDF_FOLDER, f) for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf") or f.endswith(".pdf_")]
    if not pdf_paths:
        print("No PDFs found in folder:", PDF_FOLDER)
        return

    print("Extracting text and metadata from PDFs...")
    texts, metadatas = get_pdf_text_with_metadata(pdf_paths)

    print("Splitting text into chunks...")
    text_chunks, chunk_metadatas = get_text_chunks_with_metadata(texts, metadatas)

    print("Creating vectorstore with metadata...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings, metadatas=chunk_metadatas)

    print("Saving vectorstore to", VECTORSTORE_PATH)
    with open(VECTORSTORE_PATH, "wb") as f:
        pickle.dump(vectorstore, f)

    # Verification step
    print("Sample metadata from first few chunks:")
    for i, metadata in enumerate(chunk_metadatas[:3]):
        print(f"Chunk {i+1}: {metadata}")

    print("Done! Vectorstore now includes PDF titles as metadata.")

if __name__ == "__main__":
    main()