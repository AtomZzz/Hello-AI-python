# Mini RAG

This folder contains a lightweight RAG implementation for internal knowledge retrieval.

## Files

- `knowledge.txt`: internal knowledge base text.
- `vector_store.py`: FAISS + sentence-transformers vector retrieval.

## How It Works

1. `ChatService` decides whether a user question should use RAG.
2. `RagService` retrieves top relevant chunks from `knowledge.txt`.
3. Retrieved chunks are appended to user input and sent to the selected LLM.
4. Retrieved hits are printed in console for debugging.

