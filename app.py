import os
import shutil
import uuid
import asyncio
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from faster_whisper import WhisperModel
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import edge_tts

# Configuration
OLLAMA_MODEL = "mistral"
VOICE = "en-US-ChristopherNeural"
STATIC_DIR = "static"
AUDIO_DIR = "static/audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

class WebAgent:
    def __init__(self):
        print("Initializing Web Agent components...")
        self.whisper = WhisperModel("small", device="cpu", compute_type="int8")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_db = Chroma(persist_directory="./chroma_db", embedding_function=self.embeddings)
        self.llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
        
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        Keep the answer concise and conversational.
        Context: {context}
        Question: {question}
        Answer:"""
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_db.as_retriever(),
            chain_type_kwargs={"prompt": PromptTemplate(template=prompt_template, input_variables=["context", "question"])}
        )

agent = WebAgent()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(file: UploadFile = File(...)):
    # 1. Save uploaded audio
    session_id = str(uuid.uuid4())
    input_path = os.path.join(AUDIO_DIR, f"{session_id}_input.wav")
    output_path = os.path.join(AUDIO_DIR, f"{session_id}_output.mp3")
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 2. Transcribe (VTS)
    segments, _ = agent.whisper.transcribe(input_path, beam_size=5, language="en")
    user_text = "".join([s.text for s in segments])
    
    if not user_text.strip():
        return {"user": "", "bot": "I didn't hear anything.", "audio": None}

    # 3. RAG Query
    bot_response = agent.qa_chain.run(user_text)
    
    # 4. Synthesize (STV)
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
