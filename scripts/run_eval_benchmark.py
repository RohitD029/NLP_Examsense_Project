"""
Benchmark runner for ExamSense Retrieval Evaluation
Simulates course materials and runs evaluation across Dense, BM25, and Hybrid RRF.
Outputs the exact summary table and metrics for the report.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.retrieval.vector_store import add_document, _get_store
from core.evaluation.eval_runner import run_retrieval_evaluation
import json

def main():
    store = _get_store()
    store["documents"] = {}
    store["chunks"] = []

    # Create realistic lecture chunks for a course on NLP and Deep Learning
    docs_data = [
        # Document 1: NLP_Lecture_01_Foundations.pdf
        {
            "doc_name": "NLP_Lecture_01_Foundations.pdf",
            "chunks": [
                {
                    "chunk_id": "doc1_p1_c1",
                    "text": "[NLP_Lecture_01_Foundations.pdf · Page 1] Introduction to Natural Language Processing. Tokenization is the process of segmenting text into words, subwords, or symbols. Byte-Pair Encoding (BPE) and WordPiece are common subword tokenization algorithms used to handle out-of-vocabulary words.",
                    "metadata": {"doc_name": "NLP_Lecture_01_Foundations.pdf", "page_or_slide": 1}
                },
                {
                    "chunk_id": "doc1_p2_c1",
                    "text": "[NLP_Lecture_01_Foundations.pdf · Page 2] Language Modeling and N-grams. An n-gram model predicts the next word based on the previous n-1 words using Markov assumptions. Perplexity is the standard evaluation metric for language models, defined as the exponentiated cross-entropy loss.",
                    "metadata": {"doc_name": "NLP_Lecture_01_Foundations.pdf", "page_or_slide": 2}
                },
                {
                    "chunk_id": "doc1_p3_c1",
                    "text": "[NLP_Lecture_01_Foundations.pdf · Page 3] Word Embeddings: Word2Vec and GloVe. Continuous Bag of Words (CBOW) predicts the center target word from context words, whereas Skip-Gram predicts context words from the center word using negative sampling.",
                    "metadata": {"doc_name": "NLP_Lecture_01_Foundations.pdf", "page_or_slide": 3}
                },
                {
                    "chunk_id": "doc1_p4_c1",
                    "text": "[NLP_Lecture_01_Foundations.pdf · Page 4] Sequence Labeling and Recurrent Neural Networks. Traditional RNNs suffer from vanishing and exploding gradients when processing long sequences, preventing them from learning long-distance dependencies.",
                    "metadata": {"doc_name": "NLP_Lecture_01_Foundations.pdf", "page_or_slide": 4}
                },
                {
                    "chunk_id": "doc1_p5_c1",
                    "text": "[NLP_Lecture_01_Foundations.pdf · Page 5] LSTMs and GRUs. Long Short-Term Memory networks solve vanishing gradients by introducing memory cell state and three gates: forget gate, input gate, and output gate to regulate information flow.",
                    "metadata": {"doc_name": "NLP_Lecture_01_Foundations.pdf", "page_or_slide": 5}
                },
            ]
        },
        # Document 2: Deep_Learning_Transformers.pdf
        {
            "doc_name": "Deep_Learning_Transformers.pdf",
            "chunks": [
                {
                    "chunk_id": "doc2_p1_c1",
                    "text": "[Deep_Learning_Transformers.pdf · Page 1] The Transformer Architecture. Vaswani et al. introduced the transformer relying entirely on self-attention mechanisms without recurrence or convolutions, enabling massive parallelization during training.",
                    "metadata": {"doc_name": "Deep_Learning_Transformers.pdf", "page_or_slide": 1}
                },
                {
                    "chunk_id": "doc2_p2_c1",
                    "text": "[Deep_Learning_Transformers.pdf · Page 2] Scaled Dot-Product Attention is computed as softmax(Q K^T / sqrt(d_k)) V. The scaling factor sqrt(d_k) counteracts extremely large dot products that push softmax into regions with vanishing gradients.",
                    "metadata": {"doc_name": "Deep_Learning_Transformers.pdf", "page_or_slide": 2}
                },
                {
                    "chunk_id": "doc2_p3_c1",
                    "text": "[Deep_Learning_Transformers.pdf · Page 3] Multi-Head Attention projects Queries, Keys, and Values h times with distinct learned linear projections, allowing the model to jointly attend to information from different representation subspaces.",
                    "metadata": {"doc_name": "Deep_Learning_Transformers.pdf", "page_or_slide": 3}
                },
                {
                    "chunk_id": "doc2_p4_c1",
                    "text": "[Deep_Learning_Transformers.pdf · Page 4] Positional Encodings inject token order into non-recurrent transformers using sinusoidal functions of varying frequencies: PE(pos, 2i) = sin(pos/10000^(2i/d_model)) and cosine for odd dimensions.",
                    "metadata": {"doc_name": "Deep_Learning_Transformers.pdf", "page_or_slide": 4}
                },
                {
                    "chunk_id": "doc2_p5_c1",
                    "text": "[Deep_Learning_Transformers.pdf · Page 5] BERT and Pre-training Objectives. BERT uses Masked Language Modeling (MLM) and Next Sentence Prediction (NSP) on a bidirectional transformer encoder to learn deep bidirectional representations.",
                    "metadata": {"doc_name": "Deep_Learning_Transformers.pdf", "page_or_slide": 5}
                },
            ]
        },
        # Document 3: Optimization_and_Evaluation.docx
        {
            "doc_name": "Optimization_and_Evaluation.docx",
            "chunks": [
                {
                    "chunk_id": "doc3_p1_c1",
                    "text": "[Optimization_and_Evaluation.docx · Page 1] Gradient Descent Variants. Stochastic Gradient Descent (SGD) computes noisy gradients per sample. Adam optimizer combines momentum (first moment exponential moving average) and RMSProp (second moment uncentered variance).",
                    "metadata": {"doc_name": "Optimization_and_Evaluation.docx", "page_or_slide": 1}
                },
                {
                    "chunk_id": "doc3_p2_c1",
                    "text": "[Optimization_and_Evaluation.docx · Page 2] Learning Rate Schedulers. Cosine annealing and warmup schedules prevent early divergence in deep networks and ensure stable parameter convergence near local minima.",
                    "metadata": {"doc_name": "Optimization_and_Evaluation.docx", "page_or_slide": 2}
                },
                {
                    "chunk_id": "doc3_p3_c1",
                    "text": "[Optimization_and_Evaluation.docx · Page 3] Evaluation Metrics in NLP. BLEU (Bilingual Evaluation Understudy) measures n-gram precision against human references with a brevity penalty for machine translation.",
                    "metadata": {"doc_name": "Optimization_and_Evaluation.docx", "page_or_slide": 3}
                },
                {
                    "chunk_id": "doc3_p4_c1",
                    "text": "[Optimization_and_Evaluation.docx · Page 4] ROUGE Score. Recall-Oriented Understudy for Gesting Evaluation measures n-gram overlap primarily focused on recall for text summarization tasks, including ROUGE-1, ROUGE-2, and ROUGE-L.",
                    "metadata": {"doc_name": "Optimization_and_Evaluation.docx", "page_or_slide": 4}
                },
                {
                    "chunk_id": "doc3_p5_c1",
                    "text": "[Optimization_and_Evaluation.docx · Page 5] Overfitting and Regularization. Techniques like Dropout, Layer Normalization, Weight Decay (L2 regularization), and early stopping prevent complex models from memorizing training splits.",
                    "metadata": {"doc_name": "Optimization_and_Evaluation.docx", "page_or_slide": 5}
                },
            ]
        }
    ]

    for d in docs_data:
        add_document(d["doc_name"], d["chunks"], "Ready", len(d["chunks"]))

    # 15 realistic evaluation test cases
    test_cases = [
        {"question": "How do subword tokenizers handle out of vocabulary words?", "expected_doc": "NLP_Lecture_01_Foundations.pdf", "expected_page": 1},
        {"question": "What is the standard evaluation metric for language modeling quality?", "expected_doc": "NLP_Lecture_01_Foundations.pdf", "expected_page": 2},
        {"question": "What is the core difference between Word2Vec CBOW and Skip-Gram?", "expected_doc": "NLP_Lecture_01_Foundations.pdf", "expected_page": 3},
        {"question": "Why do standard recurrent networks struggle with long distance dependencies?", "expected_doc": "NLP_Lecture_01_Foundations.pdf", "expected_page": 4},
        {"question": "What are the three gating mechanisms that control cell state in LSTM?", "expected_doc": "NLP_Lecture_01_Foundations.pdf", "expected_page": 5},
        {"question": "Why can transformers be trained in parallel compared to RNNs?", "expected_doc": "Deep_Learning_Transformers.pdf", "expected_page": 1},
        {"question": "What is the mathematical formulation and scaling factor of scaled dot product attention?", "expected_doc": "Deep_Learning_Transformers.pdf", "expected_page": 2},
        {"question": "How does multi-head attention project queries, keys, and values across representation subspaces?", "expected_doc": "Deep_Learning_Transformers.pdf", "expected_page": 3},
        {"question": "How are sinusoidal positional encodings calculated for odd and even dimensions?", "expected_doc": "Deep_Learning_Transformers.pdf", "expected_page": 4},
        {"question": "What pre-training tasks are used by BERT to learn bidirectional representations?", "expected_doc": "Deep_Learning_Transformers.pdf", "expected_page": 5},
        {"question": "How does the Adam optimizer combine first and second momentum estimates?", "expected_doc": "Optimization_and_Evaluation.docx", "expected_page": 1},
        {"question": "Why are learning rate warmup schedules used during deep model training?", "expected_doc": "Optimization_and_Evaluation.docx", "expected_page": 2},
        {"question": "How does BLEU score compute n-gram precision and brevity penalty?", "expected_doc": "Optimization_and_Evaluation.docx", "expected_page": 3},
        {"question": "What is the difference between ROUGE-1, ROUGE-2, and ROUGE-L in summarization?", "expected_doc": "Optimization_and_Evaluation.docx", "expected_page": 4},
        {"question": "What regularization methods prevent neural networks from overfitting on training data?", "expected_doc": "Optimization_and_Evaluation.docx", "expected_page": 5},
    ]

    print(f"Running retrieval evaluation with N = {len(test_cases)} cases...")
    res = run_retrieval_evaluation(test_cases, k=3)

    print("\n--- SUMMARY METRICS (k=3) ---")
    print(json.dumps(res["summary"], indent=2))

if __name__ == "__main__":
    main()
