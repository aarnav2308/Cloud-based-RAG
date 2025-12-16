**Project Architecture:**



**This project implements a Retrieval-Augmented Generation (RAG) system**

**using Pinecone for vector search and Ollama for local LLM inference.**



**High-level Structure:**



**rag\_on\_cloud/**

**├── rag/**

**│   └── rag\_chain.py        # End-to-end RAG pipeline**

**├── lambda\_query/**

**│   └── query\_retrieval.py  # Vector retrieval from Pinecone**

**├── scripts/**

**│   └── ingest.py           # Document ingestion \& embedding**

**├── requirements.txt        # Local dependencies**

**├── requirements.lambda.txt # Lambda-compatible dependencies**

**├── README.md**

**└── screenshots/**



**Execution Flow:**



**1. Documents are embedded and stored in Pinecone**

**2. User query is embedded locally via Ollama**

**3. Top-k vectors are retrieved from Pinecone**

**4. Context is injected into prompt**

**5. Ollama generates grounded response**



