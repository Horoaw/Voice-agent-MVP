import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def ingest_data():
    if not os.path.exists("data/knowledge.txt"):
        print("Error: data/knowledge.txt not found.")
        return

    print("Loading data...")
    loader = TextLoader("data/knowledge.txt")
    documents = loader.load()

    # 切分文本
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    print(f"Split into {len(texts)} chunks.")

    print("Creating vector database with Local Embeddings...")
    # 使用本地模型，无需 API Key
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 持久化存储
    db = Chroma.from_documents(
        texts, 
        embeddings, 
        persist_directory="./chroma_db"
    )
    print("Database created successfully at ./chroma_db")

if __name__ == "__main__":
    ingest_data()