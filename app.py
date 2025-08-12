import json
import faiss
import numpy as np
import yaml
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Load config
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

DOCS_DIR = Path(config["docs_dir"])
INDEX_DIR = DOCS_DIR / config["index_dir"]
MODEL_NAME = config["model"]["name"]
DEVICE = config["model"]["device"]
TOP_K = config["search"]["top_k"]

# Load model + index
print("Loading model...")
model = SentenceTransformer(MODEL_NAME, device=DEVICE)
print("Loading FAISS index...")
index = faiss.read_index(str(INDEX_DIR / "faiss_index.bin"))
with open(INDEX_DIR / "metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

app = FastAPI()

# CORS: allow your PWA origin(s)
"""
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "https://your-local-https-host"],
    allow_methods=["*"], allow_headers=["*"]
)
"""

# TODO: load your FAISS index + embedding model once here at startup

class Query(BaseModel):
    query: str
    k: int = TOP_K


@app.post("/search")
def search(q: Query):
    q_emb = model.encode(q.query)
    q_emb = np.asarray(q_emb, dtype="float32")
    q_emb /= (np.linalg.norm(q_emb) + 1e-12)
    q_emb = q_emb.reshape(1,-1)

    D, I = index.search(q_emb, q.k)
    results = []
    for idx, dist in zip(I[0], D[0]):
        if idx == -1:
            continue
        results.append({
            "score": float(dist),
            "source": metadata[idx]["source"],
            "content": metadata[idx]["content"]
        })
    return {"query": q.query, "results": results}
