# app.py
import streamlit as st
import pickle
import os
import json
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template

# -----------------------------
# Evaluation imports
# -----------------------------
try:
    from evaluate import evaluate_quick, evaluate_bleu, evaluate_rouge, evaluate_ragas_simple, evaluate_sbert_cosine
    EVALUATION_AVAILABLE = True
except Exception as e:
    st.error(f"Evaluation module import error: {e}")
    EVALUATION_AVAILABLE = False

# -----------------------------
# Constants
# -----------------------------
VECTORSTORE_PATH = "vectorstore.pkl"
PREDICTIONS_LOG = "predictions_log.json"

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

# -----------------------------
# Load vectorstore
# -----------------------------
@st.cache_resource
def load_vectorstore(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# -----------------------------
# Setup conversation chain
# -----------------------------
def get_conversation_chain(vectorstore):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found")

    llm = ChatOpenAI(
        model_name="openai/gpt-oss-20b:free",
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.5
    )

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=False,
        verbose=False
    )

# -----------------------------
# Save/load predictions
# -----------------------------
def save_predictions_log():
    # Combine all QA pairs for saving
    all_qa_pairs = st.session_state.all_qa_pairs + st.session_state.current_session_qa
    with open(PREDICTIONS_LOG, "w", encoding="utf-8") as f:
        json.dump(all_qa_pairs, f, ensure_ascii=False, indent=2)

def load_predictions_log():
    if os.path.exists(PREDICTIONS_LOG):
        with open(PREDICTIONS_LOG, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# -----------------------------
# Handle user input
# -----------------------------
def handle_userinput(user_question):
    try:
        response = st.session_state.conversation({"question": user_question})
        answer = response["answer"]
        # Add to current session for display
        st.session_state.current_session_qa.append({"question": user_question, "answer": answer})
        # Save all QA pairs (previous + current session)
        save_predictions_log()
        # Mark that we've processed this question
        st.session_state.last_processed_question = user_question
        st.session_state.should_clear_input = True  # Flag to clear input
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")

# -----------------------------
# Display current chat only (no previous conversations)
# -----------------------------
def display_current_chat():
    if st.session_state.current_session_qa:
        # Only show Q&A from the current session
        for qa in st.session_state.current_session_qa:
            question = qa.get("question") or qa.get("q") or "Unknown question"
            answer = qa.get("answer") or qa.get("a") or "No answer"
            st.write(user_template.replace("{{MSG}}", question), unsafe_allow_html=True)
            st.write(bot_template.replace("{{MSG}}", answer), unsafe_allow_html=True)
            st.markdown("---")

# -----------------------------
# Main Streamlit app
# -----------------------------
def main():
    st.set_page_config(page_title="RAG PDF Chat", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    st.header("Chat with multiple PDFs :books:")

    # Initialize session states
    if "conversation_initialized" not in st.session_state:
        st.session_state.conversation_initialized = False
    if "last_processed_question" not in st.session_state:
        st.session_state.last_processed_question = ""
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    if "current_session_qa" not in st.session_state:
        st.session_state.current_session_qa = []
    if "all_qa_pairs" not in st.session_state:
        st.session_state.all_qa_pairs = load_predictions_log()
    if "should_clear_input" not in st.session_state:
        st.session_state.should_clear_input = False

    # Load vectorstore & conversation
    if not st.session_state.conversation_initialized:
        if os.path.exists(VECTORSTORE_PATH):
            try:
                vectorstore = load_vectorstore(VECTORSTORE_PATH)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.session_state.conversation_initialized = True
            except Exception as e:
                st.error(f"Error initializing conversation: {str(e)}")
        else:
            st.error(f"Vectorstore not found at {VECTORSTORE_PATH}")
            st.info("Run create_pickle.py first")
            return

    # Display only current session chat
    display_current_chat()

    # Chat input at the bottom
    st.markdown("---")
    
    # Clear input if flag is set
    if st.session_state.should_clear_input:
        st.session_state.user_input = ""
        st.session_state.should_clear_input = False
    
    # Text input that submits on Enter
    user_question = st.text_input(
        "Ask a question:", 
        placeholder="Type your question here and press Enter...",
        key="user_input",
        label_visibility="collapsed"
    )
    
    # Process question when user presses Enter and it's a new question
    if (user_question and 
        user_question != st.session_state.last_processed_question and
        st.session_state.conversation_initialized):
        
        with st.spinner("Thinking..."):
            handle_userinput(user_question)
        st.experimental_rerun()

    # Sidebar
    with st.sidebar:
        st.header("Evaluation & About")
        st.info("Chat with your PDFs using AI.")

        total_pairs = len(st.session_state.all_qa_pairs) + len(st.session_state.current_session_qa)
        if total_pairs > 0:
            st.write(f"Total Q&A pairs: {total_pairs}")
            st.write(f"Current session: {len(st.session_state.current_session_qa)}")
            
            if st.button("Clear Current Session"):
                st.session_state.current_session_qa = []
                st.session_state.last_processed_question = ""
                st.session_state.user_input = ""
                st.session_state.should_clear_input = False
                st.experimental_rerun()

            st.markdown("---")

            if EVALUATION_AVAILABLE:
                st.subheader("Evaluation Metrics")
                if st.button("🚀 Quick Evaluation"):
                    with st.spinner("Evaluating..."):
                        # Combine all Q&A for evaluation
                        all_qa = st.session_state.all_qa_pairs + st.session_state.current_session_qa
                        results = evaluate_quick()
                    st.success("Quick Evaluation Complete!")
                    st.write(f"BLEU: {results['bleu']:.4f}")
                    st.write(f"ROUGE-1: {results['rouge']['rouge1']:.4f}")
                    st.write(f"ROUGE-2: {results['rouge']['rouge2']:.4f}")
                    st.write(f"ROUGE-L: {results['rouge']['rougeL']:.4f}")
                    st.write(f"RAGAs Faithfulness: {results['ragas_faithfulness']:.4f}")
                    st.write(f"RAGAs Answer Relevancy: {results['ragas_answer_relevancy']:.4f}")
                    st.write(f"RAGAs Answer Correctness: {results['ragas_answer_correctness']:.4f}")
                    st.write(f"RAGAs Answer Similarity: {results['ragas_answer_similarity']:.4f}")
                    st.write(f"SBERT Cosine Similarity: {results['sbert_cosine']:.4f}")
                    st.write(f"Matched pairs: {results['matched_pairs']}/{results['total_predictions']}")

if __name__ == "__main__":
    main()