from fastapi import FastAPI
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import Dict
import os
from llama_cpp import Llama

api = FastAPI()

# Load the embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

MODEL_PATH = "models/mistral-7b-instruct-v0.2.Q4_K_S.gguf"

llm = None
if os.path.exists(MODEL_PATH):
    llm = Llama(model_path=MODEL_PATH,n_ctx=2048,verbose=False)
    print("✅ LLM Loaded Successfully")
else:
    print("❌ Model file not found! Download it to 'models/'")

# Storage for indexed sites
index_storage: Dict[str, Dict] = {}

@api.post("/index_url/")
def index_url(url: str):
    """Indexes the extracted text by creating embeddings and storing them in FAISS."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        text = [p.get_text() for p in soup.find_all("p")]    
        sentences = text#.split("\n")

        embeddings = model.encode(sentences, convert_to_numpy=True)
        norm_embeddings = np.linalg.norm(embeddings,axis=1,keepdims=True)
        emb = embeddings / norm_embeddings
        d = emb.shape[1]
        faiss_index = faiss.IndexFlatIP(d)
        faiss_index.add(emb)

        index_storage[url] = {
            "faiss_index": faiss_index,
            "sentences": sentences,
            "embeddings": embeddings
        }

        return {"message": "URL indexed successfully"}
    except requests.RequestException as e:
        return {"error": f"Failed to fetch URL: {e}"}

@api.get("/ask/")
def ask(url: str, question: str):
    """Finds the most relevant sentence based on the question using FAISS."""
    if url not in index_storage:
        return {"error": "URL not indexed. Please index it first."}
    
    query_embedding = model.encode([question],convert_to_numpy=True)
    query_embedding_norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
    query_emb = query_embedding / query_embedding_norm
    faiss_index = index_storage[url]["faiss_index"]
    sentences = index_storage[url]["sentences"]
    
    cossine_similarity, I = faiss_index.search(query_emb, k=1)
    context = "\n".join(sentences[i] for i in I[0])
    prompt = f"Based exclusively on the context given answer in ONE phrase: {question}. \n Context:\n{context}"

    response = llm(prompt,max_tokens=128)['choices'][0]['text']
    
    return {"answer": response}#
            #'cossim': cossine_similarity[0][0].item(),
            #'most_similar_paragraph': sentences[I[0][0]]}
