# Formula 1 RAG Question-Answering System

A local assistant that answers race questions using real data, semantic search and Mistral 7B.

---

##  Project Overview

This project demonstrates how a Retrieval-Augmented Generation (RAG) system can be built using structured Formula 1 race data, semantic search, and a locally hosted large language model to answer questions such as “Who won the 2022 British Grand Prix?

---

##  Key Features

-  **End-to-end RAG pipeline** — from raw CSV data → preprocessed text → vector index → LLM answer
-  **Semantic retrieval** using FAISS with cosine similarity
-  **Sentence-Transformers** embeddings (`all-MiniLM-L6-v2`)
-  **Two answering modes:**
  - `rag_answer.py` -  text responses
  - `structured_qa.py` - strict JSON responses (e.g., `{"driver": "Carlos Sainz", "team": "Ferrari"}`)
-  **Local LLM integration** through LM Studio 

---

##  Installation

```bash
# 1 Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 2 Install dependencies
pip install -r requirements.txt
```

---

##  Data Setup

Download the Formula 1 dataset from [Kaggle](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020) and place all CSV files into:

```
data/raw/
```

---

##  Build the Pipeline

```bash
# Preprocess race data
python src/preprocess_data.py      # → generates data/processed/processed_races.csv

# Build FAISS vector index
python src/build_embeddings.py     # → creates models/f1_faiss.index

# Verify LM Studio connection
python src/test_lmstudio.py        # → tests local LLM integration
```

---

##  Usage

###  Text Answer Mode

Natural language responses to your Formula 1 questions:

```bash
python src/rag_answer.py --question "Who won the 2022 British Grand Prix?"
```

###  JSON Answer Mode

Structured JSON responses with specific fields:

```bash
python src/structured_qa.py --question "Who won the 2022 British Grand Prix?"
```

---
