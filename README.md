# LLM Chat App

**LLM Chat App** is a full-stack application that demonstrates how to integrate **large language models (LLMs)** into a modern web experience using a **React frontend** and a **FastAPI backend**, powered by **llama.cpp** for local inference.  

This project is designed as a **lightweight but extensible chat interface**, showing how developers can build their own custom assistants without relying on external APIs.  

## Features
- **Multi-persona Conversations** – choose from specialized roles:
  1. **Concise Email Writer** – helps craft professional and efficient messages  
  2. **Explanatory Teacher** – explains concepts with clarity and metaphors  
  3. **Technical Expert** – in-depth, technical breakdowns of complex topics  
  4. **General Chatbot** – everyday assistant for casual conversation  

- **Modern Frontend** – sleek, ChatGPT-style dark mode UI built with **React + Tailwind**  
- **Scalable Backend** – **FastAPI** routes connecting user prompts to the local LLM server  
- **Private & Local** – all inference runs through **llama.cpp**, no external API calls required  
- **Customizable** – extend with new personas, swap models, or adapt the UI for your use case  

## Tech Stack
- **Frontend:** React (Vite) + Tailwind CSS  
- **Backend:** FastAPI + Uvicorn  
- **LLM Runtime:** llama.cpp server  
- **Models:** Qwen 3 (quantized GGUF format)  

## Getting Started

### 1. Run the LLM server

[llama.cpp](https://github.com/ggml-org/llama.cpp.git)

### 2. Run the Backend
```
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### 3. Run the front end

```
cd frontend
npm install
npm run dev
```

### 4. Project File Structure

```
.
├── backend/          # FastAPI backend (API endpoints)
├── frontend/         # React frontend (chat UI)
├── models/           # Local LLM models (ignored in git)
├── llama.cpp/        # llama.cpp build + binaries (ignored in git)
└── README.md
```

## Sample Screenshot

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/4fc26744-0b07-4369-9219-c542ca16d908" />
