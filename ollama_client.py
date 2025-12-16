import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "phi3:mini"


def generate_answer(question: str, context: str) -> str:
    """
    Sends context + question to Ollama and returns the generated answer.
    """

    prompt = f"""
You are a helpful assistant.
Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
""".strip()

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()

    return response.json()["response"].strip()
