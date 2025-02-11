# Talk to URL Project - Documentation

## Overview
This project provides a **conversational interface** to interact with website content. Given a URL, the system extracts textual data, indexes it for retrieval, and allows users to ask questions about the content. A **retrieval-augmented generation (RAG)** pipeline is used to retrieve the most relevant content and generate responses using a **local LLM (Mistral 7B Instruct)**.

## Features
- **Index website content**: Extracts and stores text using FAISS for fast retrieval.
- **Conversational interface**: Enables users to ask questions about indexed content.
- **Follow-up questions memory**: Maintains session-based chat history.
- **Reranker integration**: Improves retrieval accuracy for better responses.
- **FastAPI implementation**: Provides RESTful API endpoints for seamless integration.

---

## Architecture of the Project

### **System Design**
The project follows a **modular, object-oriented architecture**, ensuring maintainability, scalability, and efficient data retrieval. The architecture consists of the following core components:

1️⃣ **Document Indexer**: 
   - Extracts text from webpages.
   - Generates sentence embeddings using **SentenceTransformers**.
   - Stores embeddings in a **FAISS** index for fast retrieval.

2️⃣ **Reranker**:
   - Enhances retrieval accuracy by re-ranking FAISS search results.
   - Uses **CrossEncoder (ms-marco-MiniLM-L-6-v2)** to improve the relevance of retrieved passages.

3️⃣ **LLM Handler**:
   - Uses **Mistral 7B Instruct** for generating responses.
   - Processes retrieved context and generates answers based on it.
   - Implements an **output parser** to clean unnecessary phrases from responses.

4️⃣ **Chat Handler**:
   - Stores and retrieves past user interactions.
   - Supports **context-aware follow-up questions** to maintain conversation flow.

5️⃣ **FastAPI Backend**:
   - Provides endpoints to index URLs, ask questions, and chat interactively.
   - Manages requests efficiently and serves API responses.

---

## Justification for Model Choices

### **1️⃣ SentenceTransformer: all-MiniLM-L6-v2**
**Why?**
- Lightweight and efficient embedding model optimized for semantic search.
- Provides **fast and accurate** sentence representations.
- Works well with **FAISS indexing** due to its compact embedding size.

### **2️⃣ FAISS for Similarity Search**
**Why?**
- Enables fast nearest-neighbor search on high-dimensional embeddings.
- Efficiently retrieves **relevant paragraphs** based on cosine similarity.
- **Scales well** for indexing large amounts of website content.

### **3️⃣ CrossEncoder (ms-marco-MiniLM-L-6-v2) for Reranking**
**Why?**
- Improves search quality by re-ranking FAISS results based on semantic meaning.
- **Significantly enhances** the relevance of retrieved paragraphs.
- Ensures the LLM receives **only the most useful context** for generating answers.

### **4️⃣ Mistral 7B Instruct as the LLM**
**Why?**
- **High-quality instruction-following model** optimized for natural language understanding.
- Performs **better than smaller models** (like TinyLlama or Phi-2) while maintaining reasonable speed.
- Supports **longer context windows** (2048 tokens), improving multi-turn interactions.

---

## Installation Guide

### **Step 1: Setup a Virtual Environment**
It is recommended to create a virtual environment before installing dependencies.
```bash
python -m venv talk_to_url_env
source talk_to_url_env/bin/activate  # On Mac/Linux
# OR
talk_to_url_env\Scripts\activate  # On Windows
```

### **Step 2: Install Dependencies**
Ensure that Python **3.11 or higher** is installed.
```bash
pip install -r requirements.txt
```

### **Step 3: Install the Local LLM**
This project uses **Mistral 7B Instruct (quantized) via llama-cpp**. To install the model, run:
```bash
python llm_install.py
```
⏳ **Installation time:** ~25-30 minutes (requires at least 4GB of disk space and 7GB+ RAM).

---

## Running the API

### **Starting the FastAPI Server**
Once the model is installed, start the API server with:
```bash
uvicorn main:api --reload
```
This runs the FastAPI application on `http://127.0.0.1:8000/`.

### **Interacting with the API**
After launching the API, you can interact using:
- **Jupyter Notebook** (`use_api.ipynb` contains examples)
- **cURL requests**
- **Postman or any HTTP client**

---

## Future Improvements
- **Optimize LLM speed** (better batching, reduce token count)
- **Deploy online** (using GPU-based cloud services)
- **Enhance retrieval with multi-query expansion**

---

## Conclusion
This project provides an efficient **RAG-based conversational interface** to interact with website content. It utilizes **FAISS retrieval, reranking, and a local LLM** to generate accurate answers. The API is **modular, efficient, and scalable** for future enhancements.

🚀 **Enjoy exploring the API!**