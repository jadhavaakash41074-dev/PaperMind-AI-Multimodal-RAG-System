from fastapi import FastAPI

import re
import numpy as np 
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer , CrossEncoder
import chromadb
from groq import Groq
import os

print("Loading RAG models... plz wait for some time")
model = SentenceTransformer("all-MiniLM-L6-v2", device = "cpu")
client = chromadb.PersistentClient(path = "./chroma_db")
collection = client.get_or_create_collection(name ="paper_30")

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
print("RAG is Loaded")

app = FastAPI()

@app.get("/")
def home():
    return {"message": "PaperMind AI is running!"}

@app.get("/ask")
def ask(question: str):
    q_emb = model.encode([question]).tolist()
    docs = collection.query(query_embeddings=q_emb, n_results=3)['documents'][0]
    context = "\n\n".join(docs)

    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer in simple words."
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    answer = response.choices[0].message.content
    return {"question": question, "answer": answer}
