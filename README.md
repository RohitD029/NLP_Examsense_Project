# 🎓 ExamSense — AI-Powered Course Material & Exam Prep Assistant

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Retrieval-Augmented Generation](https://img.shields.io/badge/NLP-Hybrid%20RAG-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**ExamSense** is an advanced, production-grade Natural Language Processing (NLP) and Retrieval-Augmented Generation (RAG) platform tailored for students and educators. It ingests course materials (lecture slides, notes, textbooks) alongside past exam question papers, computes high-yield exam topic frequencies using semantic clustering, and generates grounded, cited answers to student queries with strict hallucination guards.

---

## 🌟 Key Features

### 1. 📚 Multi-Format Document Ingestion
- Supports **PDF**, **PPTX**, **DOCX**, **EPUB**, and **TXT**.
- **OCR Fallback**: Automatically leverages **Tesseract OCR** for scanned documents or image-heavy slides.
- **Smart Chunking**: Structure-aware chunking with customizable overlap to preserve semantic continuity across headings and code blocks.

### 2. ⚡ Hybrid Retrieval (Dense + Sparse)
- **Dense Vector Search**: Powered by `BAAI/bge-small-en-v1.5` embeddings stored in high-performance **FAISS** vector indexes.
- **Sparse Keyword Search**: Full-text BM25 ranking (`rank_bm25`) for exact terminologies, acronyms, and formulas.
- **Reciprocal Rank Fusion (RRF)**: Combines dense and sparse ranking scores to deliver optimal recall and precision.

### 3. 📊 Exam Frequency & Recency Weighting
- Ingests past exam papers into an isolated collection.
- Clusters exam questions using **Agglomerative Hierarchical Clustering** on semantic embeddings.
- Applies exponential recency decay weighting so recent past papers exert higher influence in topic importance rankings.
- Visualizes recurring exam themes and marks high-yield topics directly in study answers.

### 4. 💬 Grounded Question Answering & Anti-Hallucination
- **Multi-Provider LLM Integration**: Fast inference via **Groq** (`llama-3.3-70b-versatile`, `gpt-oss-120b`) with **Google Gemini** fallback.
- **Direct Source Citations**: Every generated response includes exact document names, sections, and source chunk references.
- **Confidence Gate**: Configurable similarity threshold that instructs the model to abstain from answering if course material does not contain relevant context.

### 5. 🧪 End-to-End RAG Evaluation Suite
- In-depth benchmarking metrics:
  - **Hit@K** and **MRR@K** (Mean Reciprocal Rank)
  - **Faithfulness** & **Answer Relevance**
  - **Context Precision & Recall**
  - Retrieval and generation latency profiling

---

## 🏗️ System Architecture

```
                                  [ Upload Documents ]
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
          [ Course Materials ]                            [ Past Exam Papers ]
                    │                                             │
      ┌─────────────┴─────────────┐                 ┌─────────────┴─────────────┐
      ▼                           ▼                 ▼                           ▼
[ Text Extraction ]      [ Tesseract OCR ]    [ Text Extraction ]      [ Question Parser ]
      │                           │                 │                           │
      └─────────────┬─────────────┘                 └─────────────┬─────────────┘
                    ▼                                             ▼
             [ Token Chunking ]                          [ Agglomerative Clustering ]
                    │                                             │
       ┌────────────┴────────────┐                        [ Recency Decay Scorer ]
       ▼                         ▼                                │
[ FAISS Dense Index ]   [ BM25 Sparse Index ]                     ▼
       │                         │                       [ High-Yield Exam Topics ]
       └────────────┬────────────┘                                │
                    ▼                                             │
      [ Reciprocal Rank Fusion (RRF) ]                            │
                    │                                             │
                    ├─────────────────────────────────────────────┘
                    ▼
     [ Augmented Prompt + Citations ]
                    │
                    ▼
         [ Groq / Gemini LLM ]
                    │
                    ▼
        [ Streamlit Interface ]
```

---

## 📁 Repository Structure

```text
NLP_Examsense_Project/
├── app.py                      # Main Streamlit application entry point
├── run.bat                     # Windows one-click start script
├── check_system.bat            # Environment and dependency diagnostic script
├── requirements.txt            # Python dependencies
├── .env.example                # Example environment variable template
├── .gitignore                  # Git ignore rules for virtualenvs and secrets
├── core/
│   ├── config.py               # Pydantic-based configuration & settings
│   ├── ocr_helper.py           # Tesseract OCR extraction utility
│   ├── evaluation/             # RAG evaluation runner and metrics
│   │   ├── eval_metrics.py
│   │   └── eval_runner.py
│   ├── exam_frequency/         # Past paper question clustering and scoring
│   │   ├── frequency_scorer.py
│   │   └── question_clusterer.py
│   ├── generation/             # LLM prompt templates and API client (Groq/Gemini)
│   │   ├── llm_client.py
│   │   └── prompt_templates.py
│   ├── ingestion/              # Document parsers and chunking logic
│   │   ├── chunker.py
│   │   ├── document_parser.py
│   │   └── file_handler.py
│   └── retrieval/              # Hybrid retrieval system
│       ├── bm25_retriever.py
│       ├── embedder.py
│       ├── hybrid_retriever.py
│       └── vector_store.py
├── data/
│   ├── index/                  # Persisted FAISS and BM25 index stores (.gitkeep)
│   └── uploads/                # User document uploads directory (.gitkeep)
├── pages/
│   ├── ask.py                  # Grounded Q&A interface with source citations
│   ├── evaluation.py           # Evaluation benchmarking dashboard
│   ├── exam_topics.py          # Exam frequency analytics and charts
│   └── library.py              # Document upload and management center
├── scripts/
│   └── doctor.py               # System health and sanity check utility
└── ui/
    ├── components.py           # Reusable Streamlit UI components
    ├── safe_page.py            # Error boundary wrapper for safe page rendering
    └── styles.css              # Custom styling and design theme
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10, 3.11, or 3.12**
- **Git**
- *(Optional, for scanned documents/images)* [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed and added to your system `PATH`.

### 1. Clone the Repository

```bash
git clone https://github.com/RohitD029/NLP_Examsense_Project.git
cd NLP_Examsense_Project
```

### 2. Set Up Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your API credentials:

```bash
copy .env.example .env     # On Windows
# cp .env.example .env     # On Linux / macOS
```

Edit `.env`:
```env
# Groq API Key (Primary LLM - free from https://console.groq.com/keys)
EXAMSENSE_GROQ_API_KEY=your_groq_api_key_here

# Google Gemini API Key (Fallback LLM - free from https://aistudio.google.com/)
EXAMSENSE_GEMINI_API_KEY=your_gemini_api_key_here
```

---

## 🖥️ Running the Application

### Option A: Using the Windows Batch Script
Simply double-click or run:
```bash
run.bat
```

### Option B: Running with Streamlit
```bash
streamlit run app.py
```

Open your browser at **`http://localhost:8501`**.

---

## 💡 Usage Walkthrough

1. **Upload Documents (`/library`)**:
   - Upload course materials (slides, lecture notes, textbook chapters) under the **Course Materials** tab.
   - Upload historical exam papers under the **Past Papers** tab.
   - Click **Index Documents** to compute embeddings and build the BM25/FAISS stores.
2. **Ask Questions (`/ask`)**:
   - Ask concept queries, request code explanations, or ask for summaries.
   - View retrieved source snippets, line references, similarity scores, and topic frequency badges.
3. **Analyze Exam Trends (`/exam_topics`)**:
   - Review automatically discovered recurring questions, semantic clusters, and frequency rankings.
4. **Benchmark Performance (`/evaluation`)**:
   - Run retrieval evaluation on standard or custom test sets to inspect Hit@K, MRR, Faithfulness, and latency.

---

## 🛠️ Diagnostics & Sanity Check

To run a full diagnostic on dependencies, embedding models, vector stores, and API connectivity:

```bash
python scripts/doctor.py
```

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
