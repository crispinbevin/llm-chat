from fastapi import FastAPI, Body
from pydantic import BaseModel
import requests
from fastapi.middleware.cors import CORSMiddleware
import json, uuid, time, datetime

LOG_FILE = "chat_logs.jsonl"

def log_interaction(data: dict):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


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
    interaction_id = str(uuid.uuid4())
    start_time = time.time()
    timestamp = datetime.datetime.now().isoformat()

    system_prompt = personas.get(req.persona, personas["Normal"])

    payload = {
        "model": "Qwen3-1.7B-Q6_K.gguf",  # same name used with llama-server
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": req.message}
        ]
    }

    try:
        response = requests.post(LLAMA_SERVER, json=payload)
        response.raise_for_status()  # Raises HTTPError for bad responses
        data = response.json()

        # Extract the raw assistant reply
        raw_reply = data["choices"][0]["message"]["content"]
        cleaned_reply = clean_llm_response(raw_reply)
        
        end_time = time.time()
        response_time = end_time - start_time

        # Log successful interaction
        log_data = {
            "interaction_id": interaction_id,
            "timestamp": timestamp,
            "persona": req.persona,
            "system_prompt": system_prompt,
            "user_message": req.message,
            "raw_response": raw_reply,
            "cleaned_response": cleaned_reply,
            "response_time_seconds": response_time,
            "status": "success",
            "model": "Qwen3-1.7B-Q6_K.gguf"
        }
        log_interaction(log_data)

        return {"reply": cleaned_reply}

    except requests.exceptions.RequestException as e:
        end_time = time.time()
        response_time = end_time - start_time
        
        # Log failed interaction
        log_data = {
            "interaction_id": interaction_id,
            "timestamp": timestamp,
            "persona": req.persona,
            "system_prompt": system_prompt,
            "user_message": req.message,
            "raw_response": None,
            "cleaned_response": None,
            "response_time_seconds": response_time,
            "status": "failed",
            "error": str(e),
            "model": "Qwen3-1.7B-Q6_K.gguf"
        }
        log_interaction(log_data)
        
        return {"error": "Failed to get response from LLM server"}
    
    except Exception as e:
        end_time = time.time()
        response_time = end_time - start_time
        
        # Log unexpected errors
        log_data = {
            "interaction_id": interaction_id,
            "timestamp": timestamp,
            "persona": req.persona,
            "system_prompt": system_prompt,
            "user_message": req.message,
            "raw_response": None,
            "cleaned_response": None,
            "response_time_seconds": response_time,
            "status": "error",
            "error": str(e),
            "model": "Qwen3-1.7B-Q6_K.gguf"
        }
        log_interaction(log_data)
        
        return {"error": "Unexpected error occurred"}


    