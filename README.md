# Local Voice Agent MVP

A low-latency, high-accuracy, fully local AI Voice Agent capable of RAG (Retrieval-Augmented Generation) based Q&A.

## Key Features

*   **VTS (Voice-to-Text):** Powered by `Faster-Whisper` (local CTranslate2 implementation) for extreme speed and accuracy.
*   **Brain (RAG):** Powered by `Ollama` (Mistral/Llama2) and `LangChain` with local HuggingFace Embeddings. No OpenAI API key required.
*   **STV (Text-to-Voice):** Powered by `Edge-TTS` for high-quality, natural-sounding English speech (free).
*   **Database:** `ChromaDB` for local vector storage.

## Prerequisites

### System Dependencies (Linux)
```bash
sudo apt-get update
sudo apt-get install portaudio19-dev mpg123 git
```

### Ollama Setup
1.  Install [Ollama](https://ollama.com/).
2.  Pull the LLM model:
    ```bash
    ollama pull mistral
    ```
3.  Start the Ollama server:
    ```bash
    ollama serve
    ```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone git@github.com:Horoaw/Voice-agent-MVP.git
    cd Voice-agent-MVP
    ```

2.  **Create Conda Environment:**
    ```bash
    conda env create -f environment.yml
    conda activate vts-agent
    ```

## Usage

1.  **Ingest Knowledge Base:**
    Put your text data into `data/knowledge.txt`, then run:
    ```bash
    python ingest.py
    ```

2.  **Run the Agent:**
    ```bash
    python main.py
    ```
    - Press **Enter** to start a 5-second recording.
    - The agent will transcribe, think, and reply with audio.
    - Press **q** to quit.

## Project Structure

*   `main.py`: Core logic loop (Record -> Transcribe -> Query -> Speak).
*   `ingest.py`: Script to vectorise `data/knowledge.txt` into `chroma_db`.
*   `environment.yml`: Conda environment configuration.
*   `data/`: Directory for source documents.
