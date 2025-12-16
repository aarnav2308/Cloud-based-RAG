import json
import os
from pinecone import Pinecone

# Initialize Pinecone client
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(os.environ["PINECONE_INDEX_NAME"])


def lambda_handler(event, context):
    """
    AWS Lambda entry point for RAG retrieval
    """

    query = event.get("query", "Ecofeminism")

    response = index.query(
        vector=[0.0] * 768,  # dummy vector for now
        top_k=3,
        include_metadata=True
    )

    results = []
    for match in response["matches"]:
        results.append({
            "id": match["id"],
            "score": match["score"],
            "text": match["metadata"].get("text", "")
        })

    return {
        "statusCode": 200,
        "body": json.dumps({
            "query": query,
            "results": results
        })
    }
