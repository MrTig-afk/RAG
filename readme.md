# Chat with Multiple PDFs

A Streamlit app that allows you to chat with multiple PDF documents using vector embeddings and a large language model. The app processes PDFs, splits the text into chunks, stores them in a vector database, and answers your questions based on the uploaded documents.

---

## ✨ Features

-   **Multiple PDF Uploads**: Upload and process several PDF documents at once.
-   **Text Processing**: Automatically extracts and splits PDF text into manageable chunks for efficient analysis.
-   **Vector Storage**: Uses **FAISS** for a fast and efficient vector store.
-   **State-of-the-Art Embeddings**: Employs the `sentence-transformers/all-MiniLM-L6-v2` model for generating text embeddings.
-   **LLM Integration**: Leverages **OpenRouter's** free `openai/gpt-oss-20b:free` model for generating responses.
-   **Conversation History**: Maintains the flow of conversation using LangChain's `ConversationBufferMemory`.
-   **User-Friendly Interface**: Simple sidebar for uploading and processing your PDF files.

---

## 📋 Requirements

-   Python 3.10+
-   `pip` or a virtual environment tool like `venv`.
-   An [OpenRouter API key](https://openrouter.ai/) (the free tier supports a generous number of daily requests).

---

## 🚀 Getting Started

Follow these steps to set up and run the project on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/MrTig-afk/RAG.git
cd ask-multiple-pdfs
```

### 2. Install Dependencies

Install all the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

You need to provide your OpenRouter API key. Create a file named `.env` in the root directory of the project and add your key to it.

```bash
# .env
OPENROUTER_API_KEY="or-your-openrouter-api-key"
```

### 4. Running the App

Launch the Streamlit application with the following command:

```bash
streamlit run app.py
```

The application should now be open and running in your web browser.