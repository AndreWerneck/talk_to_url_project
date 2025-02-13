from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np
from typing import Dict
import requests

class DocumentIndexer:
    def __init__(self) -> None:
        
        """Initialize FAISS index storage and SentenceTransformer model.
        """
        
        self.index_storage: Dict[str, Dict] = {}
        self.sentence_transformer = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    def get_text(self,url: str) -> list[str] | Dict[str, str]:
        
        """Fetch and extracts text from the given URL.
        
        Args:
            url (str): The URL to fetch the text from.
        
        Returns:
            list[str]: A list of extracted text from the URL in paragraphs. It returns a dict with an error message if the URL fetching fails.
        """

        try:
            response = requests.get(url) 
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = [p.get_text() for p in soup.find_all('p')]
            return paragraphs
        
        except requests.RequestException as e:
            return {"error": f"Failed to fetch URL: {e}"}

    def index_url(self, url: str) -> Dict[str, str]:

        """Indexes the extracted text by creating embeddings and storing them in FAISS.
        
        Args:
            url (str): The URL to index the text from.
        
        Returns:
            Dict[str, str]: A message indicating the status of the indexing process.
        """

        paragraphs = self.get_text(url)
        p_embeddings = self.sentence_transformer.encode(paragraphs, convert_to_numpy=True)
        p_embeddings_norm = np.linalg.norm(p_embeddings, axis=1, keepdims=True) 
        final_p_embeddings = p_embeddings / p_embeddings_norm # normalize the paragraph embeddings
        dim = final_p_embeddings.shape[1]
        faiss_index = faiss.IndexFlatIP(dim) # Index for inner product similarity -> cosine similarity after vector normalization
        faiss_index.add(final_p_embeddings)
        
        # store the indexed data for the given url
        self.index_storage[url] = {
            "faiss_index": faiss_index,
            "paragraphs": paragraphs,
            "embeddings": final_p_embeddings
        }
        
        return {"message": "URL indexed successfully"}
    
    def retrieve_text(self, url:str, question:str, top_k:int=3,n_paragraphs_as_context:int=1) -> Dict[str,str]:
        
        """Finds the best matching paragraph based on the question using FAISS search engine based on cosine similarity.
        
        Args:
            url (str): The URL to retrieve the text from.
            question (str): The question to find the best matching paragraph.
            top_k (int): The number of top matching paragraphs to feed the reranker.
            n_paragraphs_as_context (int): The number of paragraphs to return as context.
        
        Returns:
            Dict[str, str]: The best matching paragraphs based on the question and their respective cosine similarity and relevance score with the query.
        """

        if url not in self.index_storage:
            raise ValueError("URL not indexed. Please index the URL first.")
        
        question_embedding = self.sentence_transformer.encode([question], convert_to_numpy=True)
        question_embedding_norm = np.linalg.norm(question_embedding, axis=1, keepdims=True) 
        final_question_embedding = question_embedding / question_embedding_norm # normalize the question embedding
        faiss_index = self.index_storage[url]["faiss_index"] 
        paragraphs = self.index_storage[url]["paragraphs"]

        cossine_similarity, I = faiss_index.search(final_question_embedding, k=top_k) # get the top-k similar paragraphs based on cosine similarity
        
        retrieved_paragraphs = [paragraphs[i] for i in I[0]]
        pairs = [(question,paragraph) for paragraph in retrieved_paragraphs] # create pairs of question and paragraphs for reranking
        rerank_scores = self.reranker.predict(pairs) # rerank the top-k paragraphs based on relevance score with the question

        sorted_paragraphs = [x for _, x in sorted(zip(rerank_scores, retrieved_paragraphs), reverse=True)] # sort the paragraphs based on relevance score on reverse order
        
        context = "\n".join(sorted_paragraphs[:n_paragraphs_as_context])

        return {'context':context, 'cossine_similarity':f'{cossine_similarity[0][:n_paragraphs_as_context]}', 'rerank_scores':f'{rerank_scores[:n_paragraphs_as_context]}'}