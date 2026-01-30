import os
import asyncio
import pyaudio
import wave
import time
from faster_whisper import WhisperModel
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import edge_tts
from rich.console import Console

# 初始化
console = Console()

# 配置
AUDIO_FILE = "temp_input.wav"
TTS_FILE = "temp_output.mp3"
VOICE = "en-US-ChristopherNeural" 
OLLAMA_MODEL = "mistral" # 确保你已经运行了 `ollama pull mistral`

class VTSAgent:
    def __init__(self):
        console.print("[yellow]Initializing VTS Agent (Fully Local)...[/yellow]")
        
        # 1. 加载 Whisper (VTS)
        console.print("Loading Whisper model (local)...")
        self.whisper = WhisperModel("small", device="cpu", compute_type="int8")
        
        # 2. 加载 RAG (Brain) - 本地版
        console.print("Loading Local Knowledge Base & Ollama...")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_db = Chroma(persist_directory="./chroma_db", embedding_function=self.embeddings)
        
        # 使用本地 Ollama
        self.llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
        
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Keep the answer concise and conversational.
        
        Context: {context}
        
        Question: {question}
        Answer:"""
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_db.as_retriever(),
            chain_type_kwargs={"prompt": PROMPT}
        )

    def record_audio_manual(self):
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        SECONDS = 5 # 默认录制5秒
        
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        
        console.print(f"[red]Recording for {SECONDS} seconds...[/red]")
        
        frames = []
        for i in range(0, int(RATE / CHUNK * SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
            
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        wf = wave.open(AUDIO_FILE, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

    def transcribe(self):
        start = time.time()
        console.print("Transcribing...", style="dim")
        # beam_size=5 提升精度
        segments, info = self.whisper.transcribe(AUDIO_FILE, beam_size=5, language="en")
        
        text = ""
        for segment in segments:
            text += segment.text
            
        console.print(f"[bold cyan]You said:[/bold cyan] {text} [dim]({time.time()-start:.2f}s)[/dim]")
        return text

    def query_brain(self, text):
        if not text.strip():
            return "I didn't hear anything."
        start = time.time()
        console.print("Thinking (Local LLM)...", style="dim")
        
        # 简单的异常处理，防止 Ollama 没开
        try:
            response = self.qa_chain.run(text)
        except Exception as e:
            console.print(f"[red]Error querying Ollama: {e}[/red]")
            return "Sorry, my brain is not connected. Is Ollama running?"
            
        console.print(f"[bold green]AI:[/bold green] {response} [dim]({time.time()-start:.2f}s)[/dim]")
        return response

    async def speak(self, text):
        start = time.time()
        console.print("Generating speech...", style="dim")
        communicate = edge_tts.Communicate(text, VOICE)
        await communicate.save(TTS_FILE)
        
        # 播放
        os.system(f"mpg123 {TTS_FILE} > /dev/null 2>&1") 
        
        console.print(f"[dim]Finished speaking ({time.time()-start:.2f}s)[/dim]")

async def main():
    agent = VTSAgent()
    
    while True:
        console.print("\n--- New Turn ---")
        console.print("Press [Enter] to record 5 seconds of audio, or 'q' to quit.")
        user_input = input()
        if user_input.lower() == 'q':
            break
            
        # 1. Record
        agent.record_audio_manual()
        
        # 2. Transcribe
        text = agent.transcribe()
        
        # 3. Think
        response = agent.query_brain(text)
        
        # 4. Speak
        await agent.speak(response)

if __name__ == "__main__":
    if os.system("which mpg123 > /dev/null") != 0:
        console.print("[bold red]Warning: 'mpg123' not found.[/bold red] Audio playback might fail.")
        console.print("Please install it: sudo apt-get install mpg123")
    
    asyncio.run(main())