from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from google import genai
import faiss
import numpy as np
import pickle
import os

# -----------------------
# Load Environment
# -----------------------
load_dotenv()

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)

# -----------------------
# Load FAISS Index
# -----------------------
print("Loading FAISS Index...")

index = faiss.read_index(
    "vectorstore/index.faiss"
)

# -----------------------
# Load Chunks
# -----------------------
print("Loading Chunks...")

with open(
    "vectorstore/chunks.pkl",
    "rb"
) as f:
    chunks = pickle.load(f)

print(f"Loaded {len(chunks)} chunks")

# -----------------------
# Load Embedding Model
# -----------------------
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------
# Chat Loop
# -----------------------
while True:

    query = input("\nAsk a question (type exit to quit): ")

    if query.lower() == "exit":
        break

    # Query Embedding
    query_embedding = model.encode([query])

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    # Retrieve Top Chunks
    distances, indices = index.search(
        query_embedding,
        k=3
    )

    context = ""

    for i in indices[0]:
        context += chunks[i]
        context += "\n\n"

    prompt = f"""
You are a research paper assistant.

Answer ONLY from the provided context.

If the answer is not present in the context,
say:
'I could not find this information in the uploaded paper.'

Context:
{context}

Question:
{query}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    print("\nAnswer:\n")
    print(response.text)