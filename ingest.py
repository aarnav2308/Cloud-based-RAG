import os
import boto3
import pinecone
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pathlib import Path


# -------- CONFIG --------
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "rag-index")

# Otherwise fallback to ../data
DATA_FOLDER = Path(os.environ.get("DATA_FOLDER", "../data"))
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# -------- INIT --------
from pinecone import Pinecone, Index

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Host for your index (from environment var)
index_host = os.environ.get("PINECONE_HOST")
if not index_host:
    raise ValueError("❌ Environment variable PINECONE_HOST is missing")

# Create index handle
pc_index = Index(
    name=PINECONE_INDEX_NAME,
    host=index_host,
    api_key=PINECONE_API_KEY
)



embedder = SentenceTransformer(EMBEDDING_MODEL)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

def load_pdf_text(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def process_file(filepath):
    print(f"📄 Processing file: {filepath}")
    raw_text = load_pdf_text(filepath)

    chunks = text_splitter.split_text(raw_text)
    print(f"➡️ Created {len(chunks)} chunks")

    vectors = []

    for i, chunk in enumerate(chunks):
        embedding = embedder.encode(chunk)
        vectors.append({
            "id": f"{os.path.basename(filepath)}_{i}",
            "values": embedding.tolist(),
            "metadata": {"text": chunk}
        })

    print("⬆️ Uploading to Pinecone...")
    pc_index.upsert(vectors)
    print("✅ Upload complete\n")


def main():
    print("🚀 Starting ingestion...")

    for filename in os.listdir(DATA_FOLDER):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(DATA_FOLDER, filename)
            process_file(filepath)

    print("🎉 All PDFs processed successfully!")


if __name__ == "__main__":
    main()

