# Local Semantic Search Engine

A fully local semantic search engine for Markdown, PDF, DOCX, and TXT files.
All processing — text extraction, embedding, and search — happens on your machine. No cloud.

## Core Libraries Used in this Project

- FAISS – Facebook AI Similarity Search, used for ultra-fast vector search.
- SentenceTransformers – State-of-the-art embeddings for semantic similarity.
- PyYAML – Easy configuration loading from YAML files.
- python-docx – Extracts text from Word .docx files. 
- pypdf, pypdfium2 – PDF text extraction. 
- NumPy – Vector and matrix operations. 
- FastAPI – High-performance API for searching embeddings. 
- Uvicorn – ASGI server to run the FastAPI backend.

## How It Works

1. **Collect files**  
   Recursively scans your `docs_dir` (including subfolders) for supported file types.

2. **Extract text**
    - PDF → `pypdf`
    - DOCX → `python-docx`
    - TXT / MD → plain text read  
      (Images/legacy `.doc` files not supported yet)

3. **Chunking**  
   Splits long texts into overlapping segments for better context in search.

4. **Embedding**  
   Each chunk is turned into a vector using a local [SentenceTransformers](https://www.sbert.net) model.

5. **Normalize**  
   Vectors are L2-normalized for cosine similarity.

6. **Indexing (FAISS)**  
   Stores all vectors in a FAISS `IndexFlatIP` for ultra-fast nearest-neighbor search.  
   Metadata (source path + chunk text) is stored alongside.

7. **Search**
    - Query → embed → normalize → FAISS lookup
    - Returns top matching chunks + file locations.

## Requirements
- Python 3.10+
- See `requirements.txt` for dependencies.

## Run
```bash
python build_index.py   # index your files
uvicorn app:app --host 127.0.0.1 --port 8000
Then open http://localhost:8000/docs to test.

