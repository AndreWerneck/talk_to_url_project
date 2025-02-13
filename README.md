# Talk to URL Project - Documentation

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
Ensure that Python **3.11.9 or higher** is installed.

After it, install all the depencies by making

```bash
pip install -r requirements.txt
```

### **Step 3: Install the Local LLM**
This project uses **Mistral 7B Instruct (4-bit quantized) via llama-cpp**. To install the model, run:
```bash
python3 llm_install.py
```
⏳ **Installation time:** ~25-30 minutes (requires at least 4GB of disk space and 6GB+ RAM).

**It would be easier to create a folder called "models" inside your project and store the model there.**

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
- **Jupyter Notebook** (`use_api.ipynb` contains examples and results)
- **cURL requests**

---

## API Endpoints Documentation

### **1️⃣ Index URL Content**
**Endpoint:** `POST /index_url/`
- **Description:** Extracts text from the given URL, creates embeddings, and stores them for retrieval.
- **Parameters:**
  - `url` (string, required): The webpage URL to index.
- **Example Request:**
  ```bash
  curl -X POST "http://127.0.0.1:8000/index_url/?url=https://en.wikipedia.org/wiki/Brazil"
  ```
- **Response:**
  ```json
  {"message": "URL indexed successfully"}
  ```

---

### **2️⃣ Ask a Question**
**Endpoint:** `GET /ask/`
- **Description:** Finds the most relevant content for a question using FAISS and queries the LLM.
- **Parameters:**
  - `url` (string, required): The indexed URL to retrieve text from.
  - `question` (string, required): The user’s question.
- **Example Request:**
  ```bash
  curl "http://127.0.0.1:8000/ask/?url=https://en.wikipedia.org/wiki/Brazil&question=What%20is%20the%20population%20of%20Brazil?"
  ```
- **Response:**
  ```json
  {"answer": "The population of Brazil is approximately 210.86 million as of 2022."}
  ```

---

### **3️⃣ Chat (Follow-Up Questions Support)**
**Endpoint:** `GET /chat/`
- **Description:** Enables a conversation where follow-up questions are remembered.
- **Parameters:**
  - `url` (string, required): The indexed URL.
  - `question` (string, required): The user’s question.
  - `user_id` (string, optional): The user’s unique identifier (default: `defaultuser`).
  - `max_messages_to_store` (int, optional): Number of messages to retain in memory (default: `10`).
- **Example Request:**
  ```bash
  curl "http://127.0.0.1:8000/ask/?url=https://en.wikipedia.org/wiki/Brazil&question=What%20is%20the%20population%20of%20Brazil?&user_id=user123"
  ```
- **Response:**
  ```json
  {"answer": "The population of Brazil is approximately 210.86 million as of 2022."}
  ```

- **Example Follow-up question:**
  ```bash
  curl "http://127.0.0.1:8000/ask/?url=https://en.wikipedia.org/wiki/Brazil&question=and%20how%20big%20is%20its%20territory?&user_id=user123"
  ```
- **Response:**
  ```json
  {"answer": "The territory of Brazil covers an area of approximately 8.5 million square kilometers."}
  ```

---

### **4️⃣ Get Chat History**
**Endpoint:** `GET /get_chat_history/`
- **Description:** Retrieves the stored conversation for a given user and URL.
- **Parameters:**
  - `user_id` (string, required): The user’s unique identifier.
  - `url` (string, required): The indexed URL.
- **Example Request:**
  ```bash
  curl "http://127.0.0.1:8000/get_chat_history/?user_id=user123&url=https://en.wikipedia.org/wiki/Brazil"
  ```
- **Response:**
  ```json
  {"chat_history": "User: What is the population of Brazil?\nChatbot: The population of Brazil is approximately 210.86 million as of 2022."}
  ```

---

### **5️⃣ Retrieve Text and Similarity Score**
**Endpoint:** `GET /get_retrieval_text_and_similarity/`
- **Description:** Retrieves the best matching paragraph based on the question using FAISS.
- **Parameters:**
  - `url` (string, required): The indexed URL.
  - `question` (string, required): The question to retrieve relevant text.
- **Example Request:**
  ```bash
  curl "http://127.0.0.1:8000/get_retrieval_text_and_similarity/?url=https://en.wikipedia.org/wiki/Brazil&question=What%20is%20the%20GDP%20of%20Brazil?"
  ```
- **Response:**
  ```json
  {
    "context": "Brazil's GDP was $1.8 trillion in 2023...",
    "cossine_similarity": "[0.7869417]",
    "rerank_scores": "[8.908005]"
  }
  ```
---

## Overview
This project provides a **conversational interface** to interact with website content. Given a URL, the system extracts textual data, indexes it for retrieval, and allows users to ask questions about the content. A **retrieval-augmented generation (RAG)** pipeline was chosen to be used and to retrieve the most relevant content and generate responses using a **local LLM (Mistral 7B Instruct)**.

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

### **Motivation for the Design**
The goal of this project was to create an interactive interface where users could ask questions and obtain the most relevant responses—a more intelligent alternative to a simple `Ctrl+F` search. Instead of retrieving exact phrase matches, this system understands the meaning of the question and finds the best possible answer.

To achieve this, **requests** and **BeautifulSoup4** were used to extract textual content from a given URL and process it into a readable format. The text extraction process was kept simple: only text between `<p>` and `</p>` tags was considered to ensure the retrieval of structured, meaningful content.

A crucial aspect of retrieval systems is **chunking**, which significantly influences the quality of retrieval results. Both sentence-level and paragraph-level indexing were tested. **Paragraph-level indexing was chosen** because:
- A paragraph naturally contains **more contextual information** than a single sentence.
- Retrieving whole paragraphs **preserves more meaning**, reducing the chance of missing relevant context.

Before indexing, the text is converted into embedding vectors, which aim to preserve the semantic meaning of the content. To automatically find an answer to a question, both the query and the potential responses must be transformed into vectors so that a similarity metrics can be applied. The choice of the embedding model is crucial, as automatic retrieval will only work effectively if the semantic meaning of the text is well-preserved. Several models were tested, and the selected one was **all-MiniLM-L6-v2**, specifically trained for semantic search tasks, especially for query-paragraph similarity. It delivers strong performance while maintaining an extremely fast inference speed, making it a competitive choice over more complex models. More information about **all-MiniLM-L6-v2** and other embedding models can be found in the [SentenceTransformers documentation](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html) or on the [Hugging Face model page](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

For indexing the text, **FAISS (Facebook AI Similarity Search)** was chosen as the vector database due to its efficiency and scalability. FAISS is implemented in **C++**, providing multiple optimized indexing methods that support both **CPU and GPU**, making it suitable for handling up to billions of vectors. For a deeper comparison between FAISS and other vector databases like Chroma, refer to [this detailed post](https://www.capellasolutions.com/blog/faiss-vs-chroma-lets-settle-the-vector-database-debate).

As a retrieval method, **cosine similarity** was selected, as it is the most commonly used metric for dense, high-dimensional vector similarity. More details about alternative retrieval methods and FAISS indexing strategies can be found in the [official FAISS documentation](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes).

2️⃣ **Reranker**:
   - Enhances retrieval accuracy by re-ranking FAISS search results.
   - Uses **CrossEncoder (ms-marco-MiniLM-L-6-v2)** to improve the relevance of retrieved passages.

While cosine similarity retrieval is effective, it can still return **somewhat relevant but lower-quality results**. To further improve the accuracy of retrieved passages, a **two-stage retrieval approach** was implemented:
- FAISS first retrieves the **top 3** most similar paragraphs.
- These passages are **re-ranked using a CrossEncoder model**, which evaluates their relevance more accurately.
- The passage with the highest **semantic relevance score** is selected as the final retrieved context.

The **ms-marco-MiniLM-L-6-v2 CrossEncoder** was chosen for re-ranking due to:
- Its **high retrieval precision** while maintaining fast inference speed.
- Its training on **Bing search queries**, making it highly effective for passage relevance ranking.
More details can be found [here](https://www.sbert.net/docs/pretrained-models/ce-msmarco.html) and [here](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2).

3️⃣ **LLM Handler**:
   - Uses **Mistral 7B Instruct** for generating responses.
   - Processes retrieved context and generates answers based on it.
   - Implements an **output parser** to clean unnecessary phrases from responses.

To generate the final response for the user, several strategies were considered. One possibility was to parse the text and identify patterns that could lead to answers, but this approach lacked flexibility and scalability. Instead, a LLM (Large Language Model) was integrated to receive the top-ranked paragraph from the reranker alongside the user’s question and generate a response in a customized and efficient manner. This approach added some inference time (typically between 7 to 15 seconds) but significantly improved the quality of the response.

In this implementation, the LLM is restricted to using only the retrieved context, meaning that if the answer is not found within the provided text, the model is instructed to respond with: "I don't have enough information." This decision was made because the focus of this project is more on indexing and retrieval rather than building a full-fledged RAG (Retrieval-Augmented Generation) system that blends external and pre-trained knowledge. However, in future iterations, this limitation can be lifted, allowing the model to incorporate its pre-trained knowledge when responding.

Various LLMs of different sizes were tested, including Phi-2, Gemma 2B, and TinyLlama. The best balance between performance and inference time was achieved with Mistral Instruct 7B (quantized to 4-bit), which runs smoothly on machines with at least 8GB of RAM. Thanks to the llama-cpp-python library, multiple LLMs can be downloaded and executed locally, either on CPU or GPU, with optimizations designed for diverse hardware architectures. In this case, the model is running on CPU, which allows it to function efficiently on a broad range of devices. Additionally, Mistral Instruct 7B was trained for instruction-following tasks and Q&A scenarios, making it an ideal fit for this application.

This concludes the indexing, retrieval, and response generation pipeline that powers the project.

For more information about the **llama-cpp-python** library, visit:
- [llama-cpp-python Documentation](https://llama-cpp-python.readthedocs.io/en/latest/)
- [llama.cpp GitHub Repository](https://github.com/ggerganov/llama.cpp?tab=readme-ov-file#text-only)

For details about **Mistral Instruct 7B**, refer to:
- [Mistral 7B Announcement](https://mistral.ai/en/news/announcing-mistral-7b?ref=amax.com)
- [Mistral 7B Hugging Face Model](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF)

4️⃣ **Chat Handler**:
   - Stores and retrieves past user interactions.
   - Supports **context-aware follow-up questions** to maintain conversation flow.

To enhance user experience, a **chat history feature** was added, enabling follow-up questions. This is implemented via a dictionary-based storage structure, mapping:
- **User ID → Indexed URLs → Chat Messages**.
- **Limits conversation history** to the last 10 exchanges (adjustable).

5️⃣ **FastAPI Backend**:
   - Provides endpoints to index URLs, ask questions, and chat interactively.
   - Manages requests efficiently and serves API responses.

FastAPI was chosen as the framework for implementing the API due to:
- Its **high performance** (built on Starlette and Pydantic).
- Automatic **data validation and serialization**.
- Built-in **asynchronous support** for improved request handling.

Additionally, two extra API endpoints were added:
1. **Retrieval inspection endpoint** – Allows users to examine retrieved paragraphs before they are sent to the LLM.
2. **Chat history retrieval endpoint** – Fetches stored user conversations for debugging or UX improvements.

---
## Justification for Model Choices

### **1️⃣ SentenceTransformer: all-MiniLM-L6-v2**
**Why?**
- Lightweight and efficient embedding model optimized for semantic search.
- Provides **fast and accurate** sentence representations.
- Works well with **FAISS indexing** due to its compact embedding size - 384.

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
- Performs **better than smaller models** (like TinyLlama, Gemma 2B or Phi-2) while maintaining reasonable speed.
- Supports **longer context windows** (I set to 2048 tokens, but it supports until 32768 context window tokens), improving multi-turn interactions.

---

## Conclusion
This project provides a basic but efficient **RAG-based conversational interface** to interact with website content. It utilizes **FAISS retrieval, reranking, and a local LLM** to generate accurate answers. The API is **modular trying to be as efficient and scalable as possible** for future enhancements.

🚀 **Try out the API using the `use_api.ipynb` notebook!**