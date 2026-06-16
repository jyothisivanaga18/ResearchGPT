# 📚 ResearchGPT

ResearchGPT is an AI-powered Research Paper Assistant built using Retrieval-Augmented Generation (RAG).

Users can upload research papers in PDF format and interact with them through a conversational interface.

## Features

* Upload research papers (PDF)
* Semantic document retrieval using FAISS
* Context-aware question answering
* Research paper summarization
* Key findings extraction
* Limitations analysis
* Future work generation
* Interactive Streamlit dashboard

## Tech Stack

* Python
* Streamlit
* FAISS
* Sentence Transformers
* Groq API
* Llama 3.1
* PyPDF
* LangChain Text Splitters

## Architecture

PDF → Chunking → Embeddings → FAISS → Retrieval → Llama 3.1 → Answer

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Author

Jyothi Siva Naga
