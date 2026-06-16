from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from google import genai
import faiss
import numpy as np
import os

# --------------------
# Load API Key
# --------------------
load_dotenv()

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)

# --------------------
# Read PDF
# --------------------
reader = PdfReader("data/attention.pdf")

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text

# --------------------
# Chunk Text
# --------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_text(text)

# --------------------
# Embeddings
# --------------------
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

embeddings = model.encode(chunks)

embeddings = np.array(embeddings).astype("float32")

# --------------------
# FAISS
# --------------------
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

# --------------------
# User Query
# --------------------
query = input("Ask a question: ")

query_embedding = model.encode([query])

query_embedding = np.array(query_embedding).astype("float32")

distances, indices = index.search(
    query_embedding,
    k=3
)

# --------------------
# Build Context
# --------------------
context = ""

for i in indices[0]:
    context += chunks[i]
    context += "\n\n"

# --------------------
# Prompt
# --------------------
prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question:
{query}
"""

# --------------------
# Gemini
# --------------------
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)

print("\nAnswer:\n")
print(response.text)