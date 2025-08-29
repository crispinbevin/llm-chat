from fastapi import FastAPI, Body
from pydantic import BaseModel
import requests
from fastapi.middleware.cors import CORSMiddleware

import re

def clean_llm_response(response: str) -> str:
    cleaned = re.sub(r'<think>.*?</think>\s*', '', response, flags=re.DOTALL)
    
    cleaned = cleaned.strip()
    
    cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)
    
    return cleaned

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # maybe restrict later to ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

personas = {
    "Writing": "You are a concise professional email/message writer. Help the user draft clear and polite emails or short messages. Just get to the message itself without extra fluff and ask for feedback if they want something different.",
    "Teaching": "You are an explanatory teacher. Use simple language and some metaphors to make concepts easier to understand.",
    "Expert": "You are a technically competent professional. Explain things in precise technical terms, using domain knowledge.",
    "Normal": "You are a normal chatbot. Be casual, friendly, and conversational.",
}

LLAMA_SERVER = "http://localhost:8080/v1/chat/completions"

class ChatRequest(BaseModel):
    persona: str
    message: str

@app.get("/personas")
def get_personas():
    return {"available_personas": list(personas.keys())}

@app.post("/chat")
def chat(req: ChatRequest):
    system_prompt = personas.get(req.persona, personas["Normal"])

    payload = {
        "model": "Qwen3-1.7B-Q6_K.gguf",  # same name used with llama-server
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": req.message}
        ]
    }

    response = requests.post(LLAMA_SERVER, json=payload)
    data = response.json()

    # Extract the assistant reply
    reply = data["choices"][0]["message"]["content"]

    return {"reply": clean_llm_response(reply)}
    