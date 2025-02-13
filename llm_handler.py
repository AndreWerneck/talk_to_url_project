import os
import re
from llama_cpp import Llama

class LLMHandler:
    def __init__(self, model_path: str='models/mistral-7b-instruct-v0.2.Q4_K_M.gguf', n_ctx: int=2048, verbose: bool=False) -> None:

        """Initialize the Llama model.

        Args:
            model_path (str): The path to the Llama model file. The model should be inside your 'models' directory and should be a GGUF file. Defaults to 'models/mistral-7b-instruct-v0.2.Q4_K_S.gguf'.
            n_ctx (int): The context window size. Defaults to 2048.
            verbose (bool): Whether to print verbose logs. Defaults to False.
        
        """
        if os.path.exists(model_path):
            self.llm = Llama(model_path=model_path, n_ctx=n_ctx, verbose=verbose)
            print("✅ LLM Loaded Successfully")
        else:
            raise FileNotFoundError("❌ Model file not found! Download it to 'models/'")
        
    def output_parser(self, raw_response: str) -> str:

        """Cleans up LLM output by removing unwanted characters and expressions patterns.

        Args:
            raw_response (str): The raw response generated by LLM.

        Returns:
            str: The cleaned response.
        """
        if not raw_response:
            return ""

        # Normalize spaces and remove newlines/tabs
        cleaned_response = re.sub(r'[\n\t\r]', ' ', raw_response).strip()

        # Remove unwanted expressions
        patterns_to_remove = [
            r'^\s*One phrase answer:\s*',  # Removes "One phrase answer:"
            r'^\s*answer:\s*',  # Removes "answer:"
            r'^\s*assistant:\s*',  # Removes "assistant:"
            r'^\s*AI:\s*',  # Removes "AI:"
            r'^\s*Response:\s*',  # Removes "Response:"
            r'^\s*User:\s*',  # Removes "User:"
            r'^\s*Chatbot:\s*',  # Removes "Chatbot:"
            r'^\s*System:\s*',  # Removes "System:"
            r'^\s*A:\s*',  # Removes "A:"
        ]

        # Apply regex substitutions
        for pattern in patterns_to_remove:
            cleaned_response = re.sub(pattern, '', cleaned_response, flags=re.IGNORECASE).strip()

        return cleaned_response
    
    def query_llm(self, prompt:str, max_tokens:int=128, temperature=0.0)->str:
        """Queries the LLM model with the given prompt and returns the generated response.

        Args:
            prompt (str): The prompt to query the LLM model with.
            max_tokens (int): The maximum number of tokens to generate. Defaults to 128 to avoid too long answers.
            temperature (float): The temperature value to use for sampling. Defaults to 0.0 because we want the model to base its answers mainly on the given context.

        Returns:
            str: The generated response by the LLM model cleaned by the output parser.
        """
        raw_response = self.llm(prompt,max_tokens=max_tokens,temperature=temperature)['choices'][0]['text']

        return self.output_parser(raw_response)
    