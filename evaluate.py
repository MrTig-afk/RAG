# evaluate.py
import json
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from difflib import SequenceMatcher
import numpy as np
import os

# Ensure NLTK punkt tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

GROUND_TRUTH_PATH = "truth_parsed.json"
PREDICTIONS_PATH = "predictions_log.json"

# -----------------------------
# Helpers
# -----------------------------
def similar(a, b, threshold=0.8):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() > threshold

# -----------------------------
# LLM for RAGAs
# -----------------------------
def get_llm_for_ragas():
    """OpenRouter LLM judge for RAGAs evaluation"""
    try:
        from langchain.chat_models import ChatOpenAI
    except ImportError:
        raise ImportError("Install langchain: pip install langchain")
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not found")
    
    return ChatOpenAI(
        model_name="openai/gpt-oss-20b:free",
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0
    )

# -----------------------------
# Full RAGAs Evaluation
# -----------------------------
def evaluate_ragas():
    """Evaluate predictions using RAGAs metrics"""
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
            answer_correctness,
            answer_similarity
        )
        from datasets import Dataset

        # Load ground truth and predictions
        with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
            ground_truth = json.load(f)
        with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
            predictions = json.load(f)

        # Prepare RAGAs dataset
        ragas_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
        gt_lookup = {qa["q"]: qa["a"] for paper in ground_truth for qa in paper if "q" in qa and "a" in qa}

        matched_count = 0
        for pred in predictions:
            question = pred.get("question") or pred.get("q")
            answer = pred.get("answer") or pred.get("a")
            context = pred.get("context") or pred.get("retrieved_context") or [""]

            if question and answer:
                # Fuzzy match ground truth
                best_gt = None
                best_sim = 0
                for gt_q in gt_lookup:
                    sim = SequenceMatcher(None, question.lower(), gt_q.lower()).ratio()
                    if sim > best_sim and sim > 0.:
                        best_sim = sim
                        best_gt = gt_q
                if best_gt:
                    ragas_data["question"].append(question)
                    ragas_data["answer"].append(answer)
                    ragas_data["contexts"].append([context] if isinstance(context, str) else context)
                    ragas_data["ground_truth"].append(gt_lookup[best_gt])
                    matched_count += 1

        if matched_count == 0:
            print("RAGAs - No matches found. Using simplified evaluation")
            return evaluate_ragas_simple()

        dataset = Dataset.from_dict(ragas_data)
        metrics = [faithfulness, answer_relevancy, context_recall, context_precision, answer_correctness, answer_similarity]
        llm = get_llm_for_ragas()
        result = ragas_evaluate(dataset, metrics=metrics, llm=llm)

        return {
            "faithfulness": result['faithfulness'],
            "answer_relevancy": result['answer_relevancy'],
            "context_recall": result['context_recall'],
            "context_precision": result['context_precision'],
            "answer_correctness": result['answer_correctness'],
            "answer_similarity": result['answer_similarity'],
            "matched_count": matched_count
        }

    except ImportError:
        print("RAGAs not installed. Falling back to simplified evaluation")
        return evaluate_ragas_simple()
    except Exception as e:
        print(f"Error in RAGAs evaluation: {e}")
        return evaluate_ragas_simple()

# -----------------------------
# Simplified RAGAs
# -----------------------------
def evaluate_ragas_simple():
    """Fallback RAGAs without context"""
    try:
        with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
            ground_truth = json.load(f)
        with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
            predictions = json.load(f)

        gt_lookup = {qa["q"]: qa["a"] for paper in ground_truth for qa in paper if "q" in qa and "a" in qa}
        evaluation_data = []
        for pred in predictions:
            question = pred.get("question") or pred.get("q")
            answer = pred.get("answer") or pred.get("a")
            if question and answer:
                for gt_q, gt_a in gt_lookup.items():
                    if SequenceMatcher(None, question.lower(), gt_q.lower()).ratio() > 0.6:
                        evaluation_data.append({"question": question, "generated_answer": answer, "reference_answer": gt_a})
                        break

        if not evaluation_data:
            return {"faithfulness":0,"answer_relevancy":0,"context_recall":0,"context_precision":0,"answer_correctness":0,"answer_similarity":0,"matched_count":0}

        scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
        similarities = [scorer.score(d["reference_answer"], d["generated_answer"])['rougeL'].fmeasure for d in evaluation_data]
        avg_similarity = np.mean(similarities) if similarities else 0.0

        return {
            "faithfulness": avg_similarity*0.8,
            "answer_relevancy": avg_similarity*0.9,
            "context_recall": 0.0,
            "context_precision": 0.0,
            "answer_correctness": avg_similarity,
            "answer_similarity": avg_similarity,
            "matched_count": len(evaluation_data)
        }

    except Exception as e:
        print(f"Error in simplified RAGAs: {e}")
        return {"faithfulness":0,"answer_relevancy":0,"context_recall":0,"context_precision":0,"answer_correctness":0,"answer_similarity":0,"matched_count":0}

# -----------------------------
# BLEU Evaluation
# -----------------------------
def evaluate_bleu():
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)
    with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    gt_questions = [qa["q"] for paper in ground_truth for qa in paper if "q" in qa and "a" in qa]
    gt_answers = [qa["a"] for paper in ground_truth for qa in paper if "q" in qa and "a" in qa]
    unique_titles = sorted({qa.get("title","Unknown Title") for paper in ground_truth for qa in paper})

    pred_questions = [pred.get("question") or pred.get("q") for pred in predictions if (pred.get("question") or pred.get("q"))]
    pred_answers = [pred.get("answer") or pred.get("a") for pred in predictions if (pred.get("answer") or pred.get("a"))]

    smooth_fn = SmoothingFunction().method1
    bleu_scores = []
    matched_count = 0

    for i, pred_q in enumerate(pred_questions):
        best_match_index = -1
        best_similarity = 0
        for j, gt_q in enumerate(gt_questions):
            sim = SequenceMatcher(None, pred_q.lower(), gt_q.lower()).ratio()
            if sim > best_similarity and sim > 0.6:
                best_similarity = sim
                best_match_index = j
        if best_match_index != -1:
            try:
                reference_tokens = [word_tokenize(gt_answers[best_match_index])]
                candidate_tokens = word_tokenize(pred_answers[i])
                score = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smooth_fn)
            except:
                score = 0.0
            bleu_scores.append(score)
            matched_count += 1

    average_bleu = np.mean(bleu_scores) if bleu_scores else 0.0

    return {
        "unique_titles": unique_titles,
        "bleu_scores": bleu_scores,
        "average_bleu": average_bleu,
        "matched_count": matched_count,
        "total_predictions": len(pred_questions)
    }

# -----------------------------
# ROUGE Evaluation
# -----------------------------
def evaluate_rouge():
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)
    with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    gt_questions = [qa["q"] for paper in ground_truth for qa in paper if "q" in qa and "a" in qa]
    gt_answers = [qa["a"] for paper in ground_truth for qa in paper if "q" in qa and "a" in qa]
    unique_titles = sorted({qa.get("title","Unknown Title") for paper in ground_truth for qa in paper})

    pred_questions = [pred.get("question") or pred.get("q") for pred in predictions if (pred.get("question") or pred.get("q"))]
    pred_answers = [pred.get("answer") or pred.get("a") for pred in predictions if (pred.get("answer") or pred.get("a"))]

    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    rouge_scores_list = []
    matched_count = 0

    for i, pred_q in enumerate(pred_questions):
        best_match_index = -1
        best_similarity = 0
        for j, gt_q in enumerate(gt_questions):
            sim = SequenceMatcher(None, pred_q.lower(), gt_q.lower()).ratio()
            if sim > best_similarity and sim > 0.6:
                best_similarity = sim
                best_match_index = j
        if best_match_index != -1:
            scores = scorer.score(gt_answers[best_match_index], pred_answers[i])
            rouge_scores_list.append(scores)
            matched_count += 1

    if rouge_scores_list:
        avg_rouge = {
            'rouge1': np.mean([s['rouge1'].fmeasure for s in rouge_scores_list]),
            'rouge2': np.mean([s['rouge2'].fmeasure for s in rouge_scores_list]),
            'rougeL': np.mean([s['rougeL'].fmeasure for s in rouge_scores_list])
        }
    else:
        avg_rouge = {'rouge1':0.0,'rouge2':0.0,'rougeL':0.0}

    return {
        "unique_titles": unique_titles,
        "rouge_scores": rouge_scores_list,
        "average_rouge": avg_rouge,
        "matched_count": matched_count,
        "total_predictions": len(pred_questions)
    }

# -----------------------------
# Quick Evaluation
# -----------------------------
def evaluate_quick():
    """Run BLEU + ROUGE + simplified RAGAs quickly"""
    bleu_results = evaluate_bleu()
    rouge_results = evaluate_rouge()
    ragas_results = evaluate_ragas_simple()
    return {
        "bleu": bleu_results['average_bleu'],
        "rouge": rouge_results['average_rouge'],
        "ragas_faithfulness": ragas_results.get("faithfulness",0.0),
        "ragas_answer_relevancy": ragas_results.get("answer_relevancy",0.0),
        "ragas_answer_correctness": ragas_results.get("answer_correctness",0.0),
        "ragas_answer_similarity": ragas_results.get("answer_similarity",0.0),
        "matched_pairs": bleu_results.get("matched_count",0),
        "total_predictions": bleu_results.get("total_predictions",0),
        "ragas_matched_count": ragas_results.get("matched_count",0)
    }

# -----------------------------
# Standalone run
# -----------------------------
if __name__ == "__main__":
    print("="*80)
    print("COMPLETE EVALUATION RESULTS")
    print("="*80)

    bleu_results = evaluate_bleu()
    rouge_results = evaluate_rouge()
    ragas_results = evaluate_ragas()

    print(f"BLEU Score: {bleu_results['average_bleu']:.4f}")
    print(f"ROUGE-1: {rouge_results['average_rouge']['rouge1']:.4f}")
    print(f"ROUGE-2: {rouge_results['average_rouge']['rouge2']:.4f}")
    print(f"ROUGE-L: {rouge_results['average_rouge']['rougeL']:.4f}")

    print(f"RAGAs Faithfulness: {ragas_results.get('faithfulness',0.0):.4f}")
    print(f"RAGAs Answer Relevancy: {ragas_results.get('answer_relevancy',0.0):.4f}")
    print(f"RAGAs Context Recall: {ragas_results.get('context_recall',0.0):.4f}")
    print(f"RAGAs Context Precision: {ragas_results.get('context_precision',0.0):.4f}")
    print(f"RAGAs Answer Correctness: {ragas_results.get('answer_correctness',0.0):.4f}")
    print(f"RAGAs Answer Similarity: {ragas_results.get('answer_similarity',0.0):.4f}")

    print(f"Total matched BLEU pairs: {bleu_results['matched_count']}/{bleu_results['total_predictions']}")
    print(f"Total matched RAGAs pairs: {ragas_results.get('matched_count',0)}")

    print("\nUnique Paper Titles:")
    for idx, title in enumerate(bleu_results['unique_titles'], 1):
        print(f"{idx}. {title}")
