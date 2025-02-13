from typing import Dict

class ChatHandler:
    def __init__(self)->None:
        """
        Initializes the chat history storage.
        """
        
        self.chat_history: Dict[str,Dict] = {} # chat history dict -> {user_id:{url:[messages]}}
    
    def store_message(self, user_id:str, url:str, question:str, answer:str, max_messages_to_store:int=10) -> None:
        """Stores the user's question and the chatbot's answer in the chat history per user and per URL.
        
        Args:
            user_id (str): The unique identifier of the user.
            url (str): The URL of the text document.
            question (str): The user's question.
            answer (str): The chatbot's answer.
            max_messages_to_store (int): The maximum number of messages to store in the chat history. Defaults to 10.
        """
        
        if user_id not in self.chat_history:
            self.chat_history[user_id] = {}
        if url not in self.chat_history[user_id]:
            self.chat_history[user_id][url] = []
        
        self.chat_history[user_id][url].append(f"User:{question}")
        self.chat_history[user_id][url].append(f"Chatbot:{answer}")

        # keep only the last 'max_messages_to_store' messages
        self.chat_history[user_id][url] = self.chat_history[user_id][url][-max_messages_to_store:]
    
    def get_chat_history(self, user_id:str, url:str) -> Dict[str,str]:
        """Retrieves the chat history for a given user.
        
        Args:
            user_id (str): The unique identifier of the user.
            url (str): The URL of the text document.
        
        Returns:
            Dict[str,str]: The chat history for the given user and URL.
        """
        
        if user_id in self.chat_history and url in self.chat_history[user_id]:
            return {"chat_history": "\n".join(self.chat_history[user_id][url])}
        else:
            return {"chat_history": "No chat history found for the given user and URL."}