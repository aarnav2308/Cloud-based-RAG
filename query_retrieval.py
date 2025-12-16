# query_retrieval.py
import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, Index

# --- Read environment variables ---
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_HOST = os.environ["PINECONE_HOST"]
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "rag-index-2")

# --- Initialize Pinecone client and index ---
pc = Pinecone(api_key=PINECONE_API_KEY)

index = Index(
    name=PINECONE_INDEX_NAME,
    host=PINECONE_HOST,
    api_key=PINECONE_API_KEY
)

# --- Load embedding model (same as ingestion) ---
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def retrieve(question, k=5):
    """Embed the question, query Pinecone, and return matches."""
    q_emb = embedder.encode(question).tolist()

    # ✅ Correct Pinecone v3 query API
    response = index.query(
        vector=q_emb,
        top_k=k,
        include_metadata=True
    )

    matches = response.get("matches", [])

    results = []
    for m in matches:
        results.append({
            "id": m["id"],
            "score": m.get("score"),
            "text": m["metadata"].get("text", "")
        })
    return results


def pretty_print(results):
    print(f"\nTop {len(results)} matches:")
    for i, r in enumerate(results, start=1):
        print(f"\n#{i}  ID: {r['id']}")
        print(f"Score: {r['score']}")
        print("Text snippet:")
        print(r["text"][:350].replace("\n", " "), "...")


if __name__ == "__main__":
    import sys

    question = sys.argv[1] if len(sys.argv) > 1 else "What is ecofeminism?"

    print("Question:", question)
    results = retrieve(question, k=5)
    pretty_print(results)
