# Document Retriever

A lightweight LangChain-based document retrieval app that searches a Pinecone vector store and returns answers with source citations.

## Overview

This repository contains a simple retrieval-augmented generation (RAG) pipeline built with:

- `langchain`
- `langchain_pinecone`
- `langchain_openai`
- `streamlit`

The application:

1. Embeds documents with OpenAI embeddings
2. Stores them in a Pinecone index
3. Retrieves relevant context for a user query
4. Generates an answer with citations

## Project Structure

- `main.py` - Streamlit front-end for chat and source display
- `backend/core.py` - Agent and tool integration for retrieval and answer generation
- `ingestion.py` - Document ingestion and embedding creation (if used)
- `logger.py` - Logging helpers
- `assets/` - Example screenshot images used in the README

## Setup

1. Create and activate your Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Create a `.env` file with your Pinecone and OpenAI settings:

```env
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_env
INDEX_NAME=your_index_name
```

## Running the App

```bash
- clone the repo to local
- In terminal run:
    - source .venv/bin/activate
    - uv run streamlit run main.py
```

Then open the Streamlit app in your browser if it launches automatically.

## Usage

- Enter any LangChain documentation question in the chat input.
- The app will retrieve relevant docs from Pinecone.
- It then shows an answer and lists source URLs.

## Screenshots

### App UI

<img src="assets/Screenshot 2026-06-25 at 9.52.37 AM.png" alt="App UI screenshot" width="700" />

### Retrieved Sources

<img src="assets/Screenshot 2026-06-25 at 9.53.00 AM (2).png" alt="Retrieved sources screenshot" width="700" />

### Example Answer

<img src="assets/Screenshot 2026-06-25 at 9.53.31 AM.png" alt="Example answer screenshot" width="700" />

## Notes

- If sources appear as `Unknown`, ensure the tool artifacts return documents with `metadata.source`.
- The current implementation expects the retrieval tool to return `Document` objects with `source` metadata.



