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

# ---------------------------
# Evaluation imports
# ---------------------------
try:
    from evaluate import evaluate_bleu, evaluate_rouge, evaluate_quick, evaluate_ragas
    EVALUATION_AVAILABLE = True
except Exception as e:
    st.error(f"Evaluation module import error: {e}")
    EVALUATION_AVAILABLE = False

# ---------------------------
# Constants
# ---------------------------
VECTORSTORE_PATH = "vectorstore.pkl"
PREDICTIONS_LOG = "predictions_log.json"

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()

# ---------------------------
# Load vectorstore
# ---------------------------
@st.cache_resource
def load_vectorstore(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# ---------------------------
# Setup conversation chain
# ---------------------------
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

# ---------------------------
# Save/load predictions
# ---------------------------
def save_predictions_log():
    with open(PREDICTIONS_LOG, "w", encoding="utf-8") as f:
        json.dump(st.session_state.qa_pairs, f, ensure_ascii=False, indent=2)

def load_predictions_log():
    if os.path.exists(PREDICTIONS_LOG):
        with open(PREDICTIONS_LOG, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# ---------------------------
# Handle user input
# ---------------------------
def handle_userinput(user_question):
    try:
        response = st.session_state.conversation({"question": user_question})
        answer = response["answer"]
        qa_pair = {"question": user_question, "answer": answer}
        st.session_state.qa_pairs.append(qa_pair)
        save_predictions_log()
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")

# ---------------------------
# Display chat
# ---------------------------
def display_chat_history():
    if st.session_state.qa_pairs:
        st.subheader("Conversation")
        for i, qa in enumerate(st.session_state.qa_pairs):
            question = qa.get("question") or qa.get("q") or "Unknown question"
            answer = qa.get("answer") or qa.get("a") or "No answer"
            st.write(user_template.replace("{{MSG}}", question), unsafe_allow_html=True)
            st.write(bot_template.replace("{{MSG}}", answer), unsafe_allow_html=True)
            if i < len(st.session_state.qa_pairs) - 1:
                st.markdown("---")

# ---------------------------
# Streamlit main app
# ---------------------------
def main():
    st.set_page_config(page_title="RAG Model with PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    st.header("Chat with multiple PDFs :books:")

    # Load vectorstore & conversation
    if "conversation" not in st.session_state:
        try:
            if os.path.exists(VECTORSTORE_PATH):
                vectorstore = load_vectorstore(VECTORSTORE_PATH)
                st.session_state.conversation = get_conversation_chain(vectorstore)
            else:
                st.error(f"Vectorstore not found at {VECTORSTORE_PATH}")
                st.info("Run create_pickle.py first")
                return
        except Exception as e:
            st.error(f"Error loading vectorstore: {str(e)}")
            return

    # Load previous QA
    if "qa_pairs" not in st.session_state:
        st.session_state.qa_pairs = load_predictions_log()

    # Display chat history
    display_chat_history()

    # Input for new question
    st.markdown("---")
    st.subheader("Ask a new question")
    with st.form(key="question_form", clear_on_submit=True):
        user_question = st.text_input("Enter your question:", placeholder="Type here...")
        submit_button = st.form_submit_button("Ask Question")

    if submit_button and user_question:
        with st.spinner("Thinking..."):
            handle_userinput(user_question)
        st.experimental_rerun()

    # Sidebar
    with st.sidebar:
        st.header("Evaluation & About")
        st.info("Chat with your PDF documents using AI.")

        if st.session_state.get("conversation"):
            st.success("✅ PDFs are loaded")
            if st.session_state.qa_pairs:
                st.write(f"Conversation length: {len(st.session_state.qa_pairs)} Q&A pairs")

            # Clear conversation button
            if st.button("Clear Conversation"):
                st.session_state.qa_pairs = []
                if os.path.exists(PREDICTIONS_LOG):
                    os.remove(PREDICTIONS_LOG)
                st.experimental_rerun()

            st.markdown("---")

            # Evaluation section
            if EVALUATION_AVAILABLE and st.session_state.qa_pairs:
                st.subheader("Evaluation Metrics")

                if st.button("🚀 Quick Evaluation"):
                    with st.spinner("Evaluating..."):
                        results = evaluate_quick()
                    st.success("Quick Evaluation Complete!")
                    st.write(f"BLEU: {results['bleu']:.4f}")
                    st.write(f"ROUGE-1: {results['rouge']['rouge1']:.4f}")
                    st.write(f"ROUGE-2: {results['rouge']['rouge2']:.4f}")
                    st.write(f"ROUGE-L: {results['rouge']['rougeL']:.4f}")
                    st.write(f"Matched pairs: {results['matched_pairs']}/{results['total_predictions']}")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📊 Detailed BLEU"):
                        with st.spinner("Calculating BLEU..."):
                            results = evaluate_bleu()
                        st.subheader("BLEU Evaluation Results")
                        st.write(f"Average BLEU: {results['average_bleu']:.4f}")
                        st.write(f"Matched pairs: {results['matched_count']}/{results['total_predictions']}")
                        if st.checkbox("Show detailed BLEU scores"):
                            for i, pair in enumerate(results['matched_pairs']):
                                st.write(f"Q{i+1} (similarity {pair['similarity']:.2f}): {pair['bleu_score']:.4f}")
                                st.write(f"Question: {pair['question'][:100]}...")

                with col2:
                    if st.button("📈 Detailed ROUGE"):
                        with st.spinner("Calculating ROUGE..."):
                            results = evaluate_rouge()
                        st.subheader("ROUGE Evaluation Results")
                        st.write(f"ROUGE-1: {results['average_rouge']['rouge1']:.4f}")
                        st.write(f"ROUGE-2: {results['average_rouge']['rouge2']:.4f}")
                        st.write(f"ROUGE-L: {results['average_rouge']['rougeL']:.4f}")
                        st.write(f"Matched pairs: {results['matched_count']}/{results['total_predictions']}")
                        if st.checkbox("Show detailed ROUGE scores"):
                            for i, pair in enumerate(results['matched_pairs']):
                                st.write(f"Q{i+1} (similarity {pair['similarity']:.2f}):")
                                st.write(f"ROUGE-1: {pair['rouge_scores']['rouge1'].fmeasure:.4f}")
                                st.write(f"ROUGE-2: {pair['rouge_scores']['rouge2'].fmeasure:.4f}")
                                st.write(f"ROUGE-L: {pair['rouge_scores']['rougeL'].fmeasure:.4f}")
                                st.write(f"Question: {pair['question'][:100]}...")

                if st.button("🧪 Full RAGAs Evaluation with LLM Judge"):
                    with st.spinner("Running RAGAs evaluation..."):
                        ragas_results = evaluate_ragas()
                    st.success("RAGAs Evaluation Complete!")
                    st.write(f"Faithfulness: {ragas_results['faithfulness']:.4f}")
                    st.write(f"Answer Relevancy: {ragas_results['answer_relevancy']:.4f}")
                    st.write(f"Context Recall: {ragas_results.get('context_recall', 0.0):.4f}")
                    st.write(f"Context Precision: {ragas_results.get('context_precision', 0.0):.4f}")
                    st.write(f"Answer Correctness: {ragas_results['answer_correctness']:.4f}")
                    st.write(f"Answer Similarity: {ragas_results['answer_similarity']:.4f}")
                    st.write(f"Matched pairs: {ragas_results['matched_count']}")

            # Show unique paper titles
            st.markdown("---")
            if st.button("📚 Show Paper Titles"):
                if EVALUATION_AVAILABLE:
                    results = evaluate_bleu()
                    st.subheader("Unique Paper Titles")
                    for idx, title in enumerate(results['unique_titles'], start=1):
                        st.write(f"{idx}. {title}")
                else:
                    st.warning("Evaluation currently disabled")

if __name__ == "__main__":
    main()
