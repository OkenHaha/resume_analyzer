import os
import chromadb
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize clients
nebius_client =  OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY")
)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Use a robust embedding model
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"

def get_embedding(text: str) -> list[float]:
    """Gets embedding for a single text string."""
    response = nebius_client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return response.data[0].embedding

def store_resume_chunks(resume_id: str, chunks: list[str]):
    """Stores resume chunks in ChromaDB."""
    collection = chroma_client.get_or_create_collection(name="resumes")
    
    embeddings = []
    for chunk in chunks:
        emb = get_embedding(chunk)
        embeddings.append(emb)
        
    collection.add(
        embeddings=embeddings,
        documents=chunks,
        ids=[f"{resume_id}_chunk_{i}" for i in range(len(chunks))],
        metadatas=[{"resume_id": resume_id} for _ in chunks] # Attach ID as metadata
    )

def query_semantic_matches(resume_id: str, jd_text: str, top_k: int = 5) -> list[str]:
    """Finds resume chunks that semantically match the JD requirements."""
    collection = chroma_client.get_or_create_collection(name="resumes")
    jd_embedding = get_embedding(jd_text)
    
    results = collection.query(
        query_embeddings=[jd_embedding],
        n_results=top_k,
        where={"resume_id": resume_id} # Filter by specific resume
    )
    
    return results['documents'][0]