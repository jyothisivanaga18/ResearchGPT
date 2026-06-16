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

# -----------------------
# Gemini Client
# -----------------------
client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)

# -----------------------
# Embedding Model
# -----------------------
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


def ask_question(query):

    # Load latest FAISS
    index = faiss.read_index(
        "vectorstore/index.faiss"
    )

    # Load latest chunks
    with open(
        "vectorstore/chunks.pkl",
        "rb"
    ) as f:
        chunks = pickle.load(f)

    # Create query embedding
    query_embedding = model.encode([query])

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        k=2
    )

    context = ""

    for i in indices[0]:
        context += chunks[i][:700]
        context += "\n\n"

    prompt = f"""
You are a research paper assistant.

Answer ONLY from the provided context.

If the answer is not present in the context,
reply:

I could not find this information in the uploaded paper.

Context:
{context}

Question:
{query}
"""

    try:

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text

    except Exception as e:

        return f"""
⚠️ Gemini API Error

{str(e)}
"""