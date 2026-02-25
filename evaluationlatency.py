import os
import re
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.download("punkt", quiet=True)

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

import ollama

# ========================= CONFIG =========================
LLM_MODEL = "deepseek-r1:8b"
SIM_THRESHOLD = 0.70
TOP_K = 4
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
USE_CONFUSION_MATRIX = True
# =========================================================


# ========================= LLM ============================
def ollama_llm(question, context):
    prompt = f"""
You are a helpful assistant.
Answer using ONLY the context below.
If the answer is not present, say "I don't know."

Context:
{context}

Question:
{question}
"""
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1}
    )

    return re.sub(
        r"<think>.*?</think>",
        "",
        response["message"]["content"],
        flags=re.DOTALL
    ).strip()


# ======================= PDF → FAISS ======================
def load_pdf_faiss(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/e5-base-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": TOP_K})


# ========================= RAG ============================
def rag_chain(question, retriever):
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    return ollama_llm(question, context)


# ======================== METRICS =========================
embedder = SentenceTransformer("intfloat/e5-base-v2")
rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

def cosine_sim(ref, pred):
    if not pred or "i don't know" in pred.lower():
        return 0.0
    e1 = embedder.encode([ref])
    e2 = embedder.encode([pred])
    return float(cosine_similarity(e1, e2)[0][0])

def bleu_score(ref, pred):
    smoothie = SmoothingFunction().method1
    return sentence_bleu(
        [ref.lower().split()],
        pred.lower().split(),
        smoothing_function=smoothie
    )

def rouge_score_fn(ref, pred):
    scores = rouge.score(ref, pred)
    return scores["rouge1"].fmeasure, scores["rougeL"].fmeasure


# ====================== EVALUATION ========================
def evaluate_model(pdf_path, test_data):
    retriever = load_pdf_faiss(pdf_path)

    y_true, y_pred, y_scores = [], [], []
    bleu_scores, rouge1_scores, rougeL_scores = [], [], []
    latencies = []

    for i, item in enumerate(test_data):
        question = item["question"]
        expected = item["expected"]

        start_time = time.perf_counter()
        predicted = rag_chain(question, retriever)
        end_time = time.perf_counter()

        latency = end_time - start_time
        latencies.append(latency)

        sim = cosine_sim(expected, predicted)
        bleu = bleu_score(expected, predicted)
        r1, rL = rouge_score_fn(expected, predicted)

        y_true.append(1)
        pred_label = 1 if sim >= SIM_THRESHOLD else 0
        y_pred.append(pred_label)
        y_scores.append(sim)

        bleu_scores.append(bleu)
        rouge1_scores.append(r1)
        rougeL_scores.append(rL)

        print(f"\nQ{i+1}: {question}")
        print(f"Latency   : {latency:.2f} seconds")
        print(f"Expected  : {expected}")
        print(f"Predicted : {predicted}")
        print(f"Cosine={sim:.2f} | BLEU={bleu:.2f} | ROUGE-1={r1:.2f}")

    return (
        y_true, y_pred, y_scores,
        bleu_scores, rouge1_scores, rougeL_scores,
        latencies
    )


# ===================== VISUALISATION ======================
def plot_metrics(y_true, y_pred, y_scores, latencies):
    if USE_CONFUSION_MATRIX:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Incorrect", "Correct"],
            yticklabels=["Incorrect", "Correct"]
        )
        plt.title("Confusion Matrix")
        plt.show()

        print("\nClassification Report")
        print(classification_report(
            y_true, y_pred,
            target_names=["Incorrect", "Correct"],
            zero_division=0
        ))

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()
    plt.show()

    plt.hist(latencies, bins=10)
    plt.xlabel("Latency (seconds)")
    plt.ylabel("Number of Questions")
    plt.title("End-to-End RAG Latency Distribution")
    plt.show()


# ======================== TEST DATA (20) ==================
test_data = [
    {"question": "What is the minimum CGPA required for admission into a Master’s Degree by Coursework for a candidate with a degree in a related field?",
     "expected": "The minimum CGPA required is 2.50, while applicants with CGPA between 2.00 and 2.49 may be considered subject to rigorous internal assessment."},

    {"question": "Under what academic qualifications can an undergraduate be directly admitted into a PhD programme?",
     "expected": "An undergraduate may be admitted directly into a PhD if they graduate with First Class Honours or a CGPA of at least 3.67 and obtain approval from the Centre of Studies."},

    {"question": "How long is an offer of admission valid if English or Arabic requirements are not yet satisfied?",
     "expected": "The offer remains valid for two years, after which failure to meet the language requirements results in automatic withdrawal."},

    {"question": "What must a student provide to remain employed while studying full-time?",
     "expected": "The student must submit written permission from their employer, otherwise they must register as a part-time student."},

    {"question": "What happens if forged documents are discovered after admission?",
     "expected": "The student will be dismissed from the University and IIUM may initiate legal action for forgery or fraud."},

    {"question": "When can an applicant be exempted from English language entry requirements?",
     "expected": "An applicant may be exempted if they studied or graduated in English or can demonstrate English proficiency subject to evaluation by CELPAD, CoS, and CPS."},

    {"question": "Is the introductory Bahasa Melayu course still a graduation requirement for international students?",
     "expected": "No, the introductory Bahasa Melayu course is no longer a graduation requirement following a Senate decision in August 2023."},

    {"question": "What is the minimum IELTS score for Science and Technology-based programmes?",
     "expected": "The minimum IELTS Academic score required is an overall band score of 5.0."},

    {"question": "What is the maximum leave of absence allowed without Senate endorsement?",
     "expected": "A student may apply for a maximum of two semesters of leave of absence throughout the study period without Senate endorsement."},

    {"question": "When is a student with NR status terminated from studies?",
     "expected": "A student with Not Registered status is terminated by Week 12 of the semester."},

    {"question": "How many times can a student change their mode of study?",
     "expected": "A student is permitted to change their mode of study once only during candidature."},

    {"question": "What is the maximum study period for a full-time Master’s student?",
     "expected": "The maximum study period for a full-time Master’s student is four years."},

    {"question": "What CGPA is required for postgraduate graduation?",
     "expected": "A postgraduate student must obtain a minimum CGPA of 3.00 to be eligible for graduation."},

    {"question": "What CGPA range places a student on First Probation?",
     "expected": "A student is placed on First Probation if their CGPA falls between 2.50 and 2.99."},

    {"question": "How long does a student have to complete an Incomplete grade?",
     "expected": "The requirements must be completed within four weeks of the subsequent semester."},

    {"question": "How many courses may a graduating student apply for re-sit examinations?",
     "expected": "A graduating student may apply for re-sit examinations for a maximum of two courses."},

    {"question": "By which semester must mixed-mode students be assigned a supervisor?",
     "expected": "Mixed-mode students must be assigned a supervisor by the second semester."},

    {"question": "What is the maximum word limit for a PhD thesis written in English or Arabic?",
     "expected": "The maximum word limit for a PhD thesis written in English or Arabic is 100,000 words."},

    {"question": "How many additional semesters are granted after a failed Master’s proposal defence?",
     "expected": "A Master’s student is granted two additional semesters to improve and re-defend the proposal."},

    {"question": "What is the minimum Publication Equivalence requirement for PhD by Research graduation?",
     "expected": "A PhD by Research student must achieve a minimum of 2.0 Publication Equivalence to graduate."}
]


# =========================== MAIN =========================
if __name__ == "__main__":
    pdf_path = "PG-Regulations.pdf"

    (
        y_true, y_pred, y_scores,
        bleu, rouge1, rougeL,
        latencies
    ) = evaluate_model(pdf_path, test_data)

    plot_metrics(y_true, y_pred, y_scores, latencies)

    print("\n===== LATENCY STATISTICS =====")
    print(f"Average latency : {np.mean(latencies):.2f} seconds")
    print(f"Minimum latency : {np.min(latencies):.2f} seconds")
    print(f"Maximum latency : {np.max(latencies):.2f} seconds")