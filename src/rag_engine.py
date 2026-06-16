from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import faiss
import numpy as np
import pickle
import os

# -----------------------
# Load Environment
# -----------------------
load_dotenv()

# -----------------------
# Groq Client
# -----------------------
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# -----------------------
# Embedding Model
# -----------------------
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)



# -----------------------
# Generic LLM Function
# -----------------------
def ask_llm(prompt):

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1
    )

    return response.choices[0].message.content


# -----------------------
# Question Answering
# -----------------------
def ask_question(query):

    # Load latest FAISS index
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

    # Retrieve top chunks
    distances, indices = index.search(
        query_embedding,
        k=8
    )

    # Build context
    context = ""

    for i in indices[0]:

        context += chunks[i]
        context += "\n\n"

    prompt = f"""
You are an expert research paper assistant.

Answer the user's question using ONLY the provided context.

Instructions:

- If the answer exists in the context, answer it directly.
- Be concise and accurate.
- Use bullet points when useful.
- Do NOT start with 'I could not find this information' if the answer exists.
- Only say 'I could not find this information in the uploaded paper.' when the answer is completely absent from the context.


Context:
{context}

Question:
{query}
"""

    try:

        return ask_llm(prompt)

    except Exception as e:

        return f"""
⚠️ LLM Error

{str(e)}
"""


# -----------------------
# Research Actions
# -----------------------
def research_action(action):

    with open(
        "vectorstore/chunks.pkl",
        "rb"
    ) as f:
        chunks = pickle.load(f)

    # Use beginning of paper
    context = "\n\n".join(
        chunks[:20]
    )

    prompt = f"""
You are an expert research paper analyst.

Research Paper Content:
{context}

Task:
{action}

Provide a detailed and well-structured answer.
"""

    try:

        return ask_llm(prompt)

    except Exception as e:

        return f"""
⚠️ LLM Error

{str(e)}
"""