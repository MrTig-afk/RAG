# 💬 Chat with Multiple PDFs (RAG Application)

A powerful Streamlit application that enables conversation with multiple PDF documents using a **Retrieval-Augmented Generation (RAG)** pipeline. This app processes your documents, stores them in a vector database, and uses a Large Language Model (LLM) to answer your questions based _only_ on the provided context.

---

## ✨ Features

- **Multiple PDF Uploads**: Upload and process several PDF documents at once via a simple sidebar interface.
- **Text Processing**: Automatically extracts and splits PDF text into manageable chunks for efficient analysis.
- **Vector Storage**: Utilizes **FAISS** (Facebook AI Similarity Search) for a fast and efficient vector store.
- **State-of-the-Art Embeddings**: Employs the highly efficient `sentence-transformers/all-MiniLM-L6-v2` model for generating text embeddings.
- **LLM Integration**: Leverages **OpenRouter's** free `openai/gpt-oss-20b:free` model for generating accurate, context-aware responses.
- **Conversation History**: Maintains the flow of conversation using LangChain's `ConversationBufferMemory`.
- **Evaluation Metrics**: Supports **BLEU**, **ROUGE**, and **RAGAs** metrics to comprehensively assess model responses and the RAG pipeline quality.

---

## 📂 Project Structure

├── app.py # Streamlit frontend (main application)
├── create_pickle.py # PDF processing and vectorstore creation script
├── evaluate.py # Evaluation scripts (BLEU, ROUGE, RAGAs)
├── htmlTemplates.py # HTML templates for the chat interface
├── vectorstore.pkl # Serialized FAISS vector store (generated)
├── predictions_log.json # Logs user questions and model answers (generated)
├── requirements.txt # Python dependencies
├── .env # Environment variables (API keys)
└── README.md # Project documentation

## ## 📋 Requirements

- **Python 3.10+**
- `pip` or a virtual environment tool like `venv`.
- An [OpenRouter API key](https://openrouter.ai/) is required for LLM access.

---

## 🚀 Getting Started

Follow these steps to set up and run the project on your local machine.

### 1. Clone the Repository

Navigate to your desired directory and clone the project:

```bash
git clone [https://github.com/MrTig-afk/RAG.git](https://github.com/MrTig-afk/RAG.git)
cd RAG
```

### 2. Install Dependencies

Install all the required Python packages:

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Create a file named .env in the root directory of the project and add your OpenRouter API key:

```bash
# .env
OPENROUTER_API_KEY="or-your-openrouter-api-key"
```

### 4. Create Vector Store (Process PDFs)

Place your PDF documents in a designated folder (e.g., a docs/ directory or the root of the project). Then run the script to process the PDFs and create the vector store:

```bash
python create_pickle.py
```

This command reads the PDFs, chunks the text, creates embeddings, and saves the FAISS index to vectorstore.pkl.

### 5. Running the App

Launch the Streamlit application:

```bash
streamlit run app.py
```

### 🛠 Usage & Evaluation

## Usage

**Chat**: Ask questions in the main input field. The app uses the RAG pipeline to retrieve context from vectorstore.pkl and generate an answer.

**Clear**: Use the sidebar option to clear the conversation history (Q&A pairs are automatically logged to predictions_log.json).

### Evaluation

The app supports running quality checks on the RAG pipeline via the sidebar:

Metric Description
BLEU Measures n-gram overlap between the predicted answer and a ground-truth reference.
ROUGE Measures recall, precision, and F1 score based on overlapping sequences (useful for summarization/paraphrasing).
RAGAs Uses an LLM judge to evaluate RAG-specific metrics like faithfulness, answer relevancy, and context correctness.

### 📚 References

- [Streamlit Documentation](https://docs.streamlit.io/)
- [LangChain Documentation](https://docs.langchain.com/)
- [OpenRouter API](https://openrouter.ai/docs)
- [Sentence Transformers](https://www.sbert.net/)
- [FAISS: Facebook AI Similarity Search](https://github.com/facebookresearch/faiss)

## 👨‍💻 Author

- **MrTig-afk**
- [GitHub Profile](https://github.com/MrTig-afk)
