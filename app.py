import os
import uuid
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import edge_tts

# RAG Imports
try:
    from langchain_ollama import ChatOllama
except ImportError:
    from langchain_community.chat_models import ChatOllama

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from langchain_core.prompts import ChatPromptTemplate

# Configuration
OLLAMA_MODEL = "mistral"
VOICE = "en-US-ChristopherNeural"
AUDIO_DIR = "static/audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

class WebAgent:
    def __init__(self):
        print("Initializing Web Agent (Text-Input Mode)...")
        # Removed Whisper (Client side handles VTS now)
        
        # Brain (CPU Forced for stability on GTX 1060)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.vector_db = Chroma(persist_directory="./chroma_db", embedding_function=self.embeddings)
        self.retriever = self.vector_db.as_retriever(search_kwargs={"k": 3})
        self.llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
        
        self.prompt_template = ChatPromptTemplate.from_template(
            "Use the following pieces of context to answer the question at the end. "
            "If the answer is not in the context, just say you don't know.\n\n"
            "Context: {context}\n\n"
            "Question: {question}\n"
            "Answer:"
        )

    def ask(self, query):
        docs = self.retriever.invoke(query)
        context_text = "\n\n".join([doc.page_content for doc in docs])
        prompt_value = self.prompt_template.invoke({"context": context_text, "question": query})
        response = self.llm.invoke(prompt_value)
        if hasattr(response, "content"):
            return response.content
        return str(response)

agent = WebAgent()

class ChatRequest(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(request: ChatRequest):
    user_text = request.text
    session_id = str(uuid.uuid4())
    output_path = os.path.join(AUDIO_DIR, f"{session_id}_output.mp3")
    
    print(f"User asking: {user_text}")

    # RAG
    bot_response = agent.ask(user_text)
    
    # STV
    communicate = edge_tts.Communicate(bot_response, VOICE)
    await communicate.save(output_path)
    
    return {
        "user": user_text,
        "bot": bot_response,
        "audio": f"/static/audio/{session_id}_output.mp3"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6969)