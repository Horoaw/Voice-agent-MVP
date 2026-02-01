import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def ingest_data():
    if not os.path.exists("data/knowledge.txt"):
        print("Error: data/knowledge.txt not found.")
        return

    print("Loading data...")
    loader = TextLoader("data/knowledge.txt")
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    print(f"Split into {len(texts)} chunks.")
    print("Creating vector database with Local Embeddings (LangChain 1.x)...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    db = Chroma.from_documents(
        texts, 
        embeddings, 
        persist_directory="./chroma_db"
    )
    print("Database created successfully at ./chroma_db")

if __name__ == "__main__":
    ingest_data()
