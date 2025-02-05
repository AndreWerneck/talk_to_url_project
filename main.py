from fastapi import FastAPI
import requests
from bs4 import BeautifulSoup

api = FastAPI()

@api.get("/fetch_url/")
def fetch_url(url : str = 'https://en.wikipedia.org/wiki/Brazil'):
    """
    
    Fetches the raw HTML content of a given URL.
    
    """
    
    try:
        response = requests.get(url)
        response.raise_for_status() # raise an error if status different of success (200) instead of return an error code
        return {"url":url,"content":response.text[:5000]}
    
    except requests.RequestException as e:
        return {"error":f"failed to fetch URL : {e}"}

@api.get("/extract_text/")
def extract_text(url:str = 'https://en.wikipedia.org/wiki/Brazil'):
    """
    
    Fetches the raw content and parse it to extract just the readable information -> withtout HTML
    
    """
    try:
        response = requests.get(url)
        response.raise_for_status() # raise an error if status different of success (200) instead of return an error code
        bsoup = BeautifulSoup(response.text, "html.parser")
        text_in_paragraphs = bsoup.find_all('p')
        extracted_text = "\n".join([paragraph.get_text() for paragraph in text_in_paragraphs])
        return {"url":url,"text":extracted_text[:5000]}
    except requests.RequestException as e:
        return {"error":f"failed to fetch URL : {e}"}
