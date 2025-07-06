import os
import PyPDF2 # type: ignore
import nltk
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer # type: ignore
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
from typing import List, Dict
import requests
import json
import logging
from nltk.tokenize import sent_tokenize
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger.info(f"OpenRouter API key loaded: {os.getenv('OPENROUTER_API_KEY') is not None}")

nltk.download('punkt')

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
collection = chroma_client.get_or_create_collection(name="medicare_docs")

def extract_text_from_pdf(pdf_path: str) -> List[Dict]:
    logger.info("Extracting text from PDF")
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        return [{"page": i + 1, "text": reader.pages[i].extract_text()} for i in range(len(reader.pages))]

# Chunk text semantically
def dynamic_chunk_text(text: str, min_chunk_size: int = 100, max_chunk_size: int = 500) -> List[str]:
    sentences = sent_tokenize(text)
    embeddings = [embedding_model.encode(s) for s in sentences]
    chunks, current_chunk, current_length = [], [], 0

    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    for i, sentence in enumerate(sentences):
        current_length += len(sentence)
        current_chunk.append(sentence)

        if current_length > max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk, current_length = [], 0
        elif i < len(sentences) - 1 and cosine_similarity(embeddings[i], embeddings[i + 1]) < 0.5 and current_length >= min_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk, current_length = [], 0

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    logger.info(f"Chunked into {len(chunks)} segments")
    return chunks

# Store chunks in ChromaDB
def store_chunks_in_db(pages: List[Dict]):
    existing_ids = set(collection.get()['ids'])
    for page in pages:
        page_prefix = f"page_{page['page']}_chunk_"
        if any(x.startswith(page_prefix) for x in existing_ids):
            continue
        chunks = dynamic_chunk_text(page["text"])
        for i, chunk in enumerate(chunks):
            collection.add(
                documents=[chunk],
                metadatas=[{"page": page["page"], "chunk_id": i}],
                ids=[f"{page_prefix}{i}"]
            )

# Query Mistral via OpenRouter
def query_mistral_openrouter(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mistralai/mistral-small-3.2-24b-instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 512,
        "temperature": 0.7
    }
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# Retrieve chunks and generate response
def retrieve_and_generate(query: str) -> Dict:
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=3)

    if not results["documents"]:
        raise HTTPException(status_code=404, detail="No relevant information found")

    context = "\n".join(results["documents"][0])
    page_numbers = [meta["page"] for meta in results["metadatas"][0]]
    chunk_sizes = [len(doc) for doc in results["documents"][0]]

    prompt = f"""
    Based on the following context, answer the query in structured JSON format.
    Please respond ONLY with a valid JSON object, no additional text.

    Query: {query}
    Context:
    {context}

    Format:
    ```json
    {{
      "answer": "Your answer here",
      "source_page": {page_numbers[0]},
      "confidence_score": 0.0,
      "chunk_size": {chunk_sizes[0]}
    }}
    """
    try:
        response_text = query_mistral_openrouter(prompt)
    except Exception as e:
        logger.error(f"Error querying Mistral via OpenRouter: {e}")
        return {
            "answer": f"Error querying Mistral: {str(e)}",
            "source_page": page_numbers[0],
            "confidence_score": 0.0,
            "chunk_size": chunk_sizes[0]
        }

    # Clean up model response: remove code block markers and leading 'json' if present
    cleaned_response = response_text.strip()
    if cleaned_response.startswith('```json'):
        cleaned_response = cleaned_response[7:]
    if cleaned_response.startswith('```'):
        cleaned_response = cleaned_response[3:]
    if cleaned_response.endswith('```'):
        cleaned_response = cleaned_response[:-3]
    cleaned_response = cleaned_response.strip()
    if cleaned_response.lower().startswith('json'):
        cleaned_response = cleaned_response[4:].strip()
    try:
        response_json = json.loads(cleaned_response)
        response_json["confidence_score"] = float(np.mean(results["distances"][0]))
        return response_json
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse JSON. Raw response:\n{response_text}")
        return {
            "answer": response_text.strip() or "Empty response from model.",
            "source_page": page_numbers[0],
            "confidence_score": float(np.mean(results["distances"][0])),
            "chunk_size": chunk_sizes[0]
        }
# HTML UI route
@app.get("/", response_class=HTMLResponse)
async def get_ui():
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())

# Query API endpoint
@app.post("/query")
async def process_query(request: QueryRequest):
    try:
        return retrieve_and_generate(request.query)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Load PDF and store embeddings
PDF_PATH = "10050-medicare-and-you.pdf"
if os.path.exists(PDF_PATH):
    logger.info("Loading and processing PDF...")
    pages = extract_text_from_pdf(PDF_PATH)
    store_chunks_in_db(pages)
else:
    logger.warning("PDF file not found.")

# Run FastAPI server
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
