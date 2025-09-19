from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel
import requests
from fastapi.middleware.cors import CORSMiddleware
import json, uuid, time, datetime
import subprocess
import sys
import os
from typing import List, Dict, Optional

# Import the crawler (make sure the crawler file is in the same directory)
try:
    from crawler import WebCrawler, RAGQueryEngine
    CRAWLER_AVAILABLE = True
except ImportError:
    CRAWLER_AVAILABLE = False
    print("Warning: Crawler module not found. Crawler endpoints will be disabled.")

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
    "RAG": "You are a helpful assistant with access to specific website information. Use the provided context to answer questions accurately. Always cite the source URLs when referencing specific information. If the context doesn't contain relevant information, say so clearly."
}

LLAMA_SERVER = "http://localhost:8080/v1/chat/completions"

# Initialize RAG query engine
rag_engine = None
if CRAWLER_AVAILABLE:
    try:
        if os.path.exists("crawl_data/metadata.json"):
            rag_engine = RAGQueryEngine("crawl_data")
            print("RAG query engine initialized successfully")
    except Exception as e:
        print(f"Failed to initialize RAG engine: {e}")

class ChatRequest(BaseModel):
    persona: str
    message: str

class CrawlRequest(BaseModel):
    url: str
    max_pages: Optional[int] = 50
    delay: Optional[float] = 1.5

class RAGQueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

@app.get("/personas")
def get_personas():
    available_personas = list(personas.keys())
    # Only include RAG persona if the engine is available
    if not rag_engine:
        available_personas = [p for p in available_personas if p != "RAG"]
    return {"available_personas": available_personas}

@app.post("/chat")
def chat(req: ChatRequest):
    interaction_id = str(uuid.uuid4())
    start_time = time.time()
    timestamp = datetime.datetime.now().isoformat()

    system_prompt = personas.get(req.persona, personas["Normal"])
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": req.message}
    ]

    # Handle RAG queries
    context = ""
    if req.persona == "RAG" and rag_engine:
        try:
            # Search for relevant context
            search_results = rag_engine.search(req.message, top_k=3)
            
            if search_results:
                context_parts = []
                urls_cited = set()
                
                for result in search_results:
                    context_parts.append(f"Content: {result['content']}")
                    urls_cited.add(result['url'])
                
                context = "\n\n".join(context_parts)
                urls_list = "\n".join([f"- {url}" for url in urls_cited])
                
                # Enhanced system prompt for RAG
                rag_system_prompt = f"""You are a helpful assistant with access to specific website information. Use the provided context to answer questions accurately. Always cite the source URLs when referencing specific information.

CONTEXT FROM WEBSITE:
{context}

SOURCE URLS:
{urls_list}

Answer the user's question using this context. If the context doesn't fully answer the question, say so clearly."""

                messages = [
                    {"role": "system", "content": rag_system_prompt},
                    {"role": "user", "content": req.message}
                ]
            else:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant. The user asked about a website, but no relevant information was found in the database. Let them know that no relevant information was found and suggest they might need to crawl the website first."},
                    {"role": "user", "content": req.message}
                ]
        except Exception as e:
            print(f"RAG search error: {e}")

    payload = {
        "model": "Qwen3-1.7B-Q6_K.gguf",
        "messages": messages
    }

    try:
        response = requests.post(LLAMA_SERVER, json=payload)
        response.raise_for_status()
        data = response.json()

        raw_reply = data["choices"][0]["message"]["content"]
        cleaned_reply = clean_llm_response(raw_reply)
        
        end_time = time.time()
        response_time = end_time - start_time

        log_data = {
            "interaction_id": interaction_id,
            "timestamp": timestamp,
            "persona": req.persona,
            "system_prompt": system_prompt,
            "user_message": req.message,
            "context_provided": bool(context),
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
        
        log_data = {
            "interaction_id": interaction_id,
            "timestamp": timestamp,
            "persona": req.persona,
            "system_prompt": system_prompt,
            "user_message": req.message,
            "context_provided": bool(context),
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
        
        log_data = {
            "interaction_id": interaction_id,
            "timestamp": timestamp,
            "persona": req.persona,
            "system_prompt": system_prompt,
            "user_message": req.message,
            "context_provided": bool(context),
            "raw_response": None,
            "cleaned_response": None,
            "response_time_seconds": response_time,
            "status": "error",
            "error": str(e),
            "model": "Qwen3-1.7B-Q6_K.gguf"
        }
        log_interaction(log_data)
        
        return {"error": "Unexpected error occurred"}

# New crawler endpoints
@app.post("/crawl")
def crawl_website(req: CrawlRequest):
    """Trigger website crawling"""
    if not CRAWLER_AVAILABLE:
        raise HTTPException(status_code=500, detail="Crawler module not available")
    
    try:
        print(f"Starting crawl of {req.url}")
        
        # Create crawler instance
        crawler = WebCrawler(
            base_url=req.url, 
            max_pages=req.max_pages, 
            delay=req.delay
        )
        
        # Run crawl in background (for production, consider using Celery or similar)
        crawler.run_full_crawl()
        
        # Reinitialize RAG engine with new data
        global rag_engine
        rag_engine = RAGQueryEngine("crawl_data")
        
        return {
            "status": "completed",
            "message": f"Successfully crawled {len(crawler.crawled_data)} chunks from {len(crawler.visited_urls)} pages",
            "total_chunks": len(crawler.crawled_data),
            "total_pages": len(crawler.visited_urls)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Crawling failed: {str(e)}")

@app.get("/crawl/status")
def get_crawl_status():
    """Get current crawl data status"""
    try:
        if os.path.exists("crawl_data/crawl_info.json"):
            with open("crawl_data/crawl_info.json", 'r') as f:
                info = json.load(f)
            return {
                "status": "data_available",
                "info": info,
                "rag_engine_ready": rag_engine is not None
            }
        else:
            return {
                "status": "no_data",
                "message": "No crawled data found. Run /crawl first.",
                "rag_engine_ready": False
            }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "rag_engine_ready": False
        }

@app.post("/search")
def search_crawled_data(req: RAGQueryRequest):
    """Search through crawled data"""
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG engine not initialized. Crawl a website first.")
    
    try:
        results = rag_engine.search(req.query, req.top_k)
        return {
            "query": req.query,
            "results": results,
            "total_found": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.delete("/crawl/data")
def clear_crawl_data():
    """Clear all crawled data"""
    try:
        import shutil
        if os.path.exists("crawl_data"):
            shutil.rmtree("crawl_data")
        
        global rag_engine
        rag_engine = None
        
        return {"status": "success", "message": "All crawled data cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear data: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)