import requests
from config import LLM_URL

class OllamaLLM:
    def __init__(self, model="llama3.2"):
        self.model = model

    def generate(self, prompt):
        url = LLM_URL

        payload = {
            "model": self.model,
           "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }

        response = requests.post(url, json=payload)

        data = response.json()

        if "message" not in data:
            raise Exception(f"Ollama error: {data}")

        return data["message"]["content"]