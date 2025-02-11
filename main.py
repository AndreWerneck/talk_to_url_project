from fastapi import FastAPI
from document_indexer import DocumentIndexer
from llm_handler import LLMHandler
from chat_handler import ChatHandler
from typing import Dict

document_indexer = DocumentIndexer()
llm_handler = LLMHandler()
chat_handler = ChatHandler()

api = FastAPI()

@api.post("/index_url/")
def index_url(url: str) -> Dict[str, str]:
    """Indexes the extracted text by creating embeddings and storing them in FAISS."""
    return document_indexer.index_url(url)

@api.get("/ask/")
def ask(url: str, question: str) -> Dict[str, str]:
    """The chatbot's response to the user's question."""
    
    retrieval_dict = document_indexer.retrieve_text(url, question)

    prompt = f"""Based only on the context given, answer in ONE phrase: {question}.
    
    If the answer is not in the context, please respond with 'I don't have enough information'.
    
    Context:\n{retrieval_dict['context']}
    """

    response = llm_handler.query_llm(prompt)

    return {"answer" : response}

@api.get("/chat/")
def chat(url:str, question:str, user_id:str = 'defaultuser', max_messages_to_store:int=10) -> Dict[str, str]:
    """Handle follow-up questions using chat memory"""

    retrieval_dict = document_indexer.retrieve_text(url, question)

    chat_history = chat_handler.get_chat_history(user_id, url)

    past_conversation = chat_history['chat_history']

    prompt = f"""Based only on both the context and the past conversation given, answer in ONE phrase: {question}.
    
    If the answer is not in the context or in the past conversation, please respond with 'I don't have enough information'.
    
    Context:\n{retrieval_dict['context']}

    Past conversation:\n{past_conversation}
    """

    response = llm_handler.query_llm(prompt)

    chat_handler.store_message(user_id, url, question, response, max_messages_to_store)

    return {"answer" : response}

@api.get("/get_chat_history/")
def get_chat_history(user_id:str, url:str) -> Dict[str, str]:
    """Retrieves the chat history for a given user and url."""
    return chat_handler.get_chat_history(user_id, url)

@api.get("/get_retrieval_text_and_similarity/")
def get_retrieval_text_and_similarity(url:str, question:str) -> Dict[str, str]:
    """Retrieves the best matching paragraph based on the question using FAISS."""
    return document_indexer.retrieve_text(url, question)
    