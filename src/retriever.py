from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

print("Loading FAISS Index...")

index = faiss.read_index(
    "vectorstore/index.faiss"
)

print("Loading Chunks...")

with open(
    "vectorstore/chunks.pkl",
    "rb"
) as f:
    chunks = pickle.load(f)

print(f"Loaded {len(chunks)} chunks")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

while True:

    query = input("\nAsk a question (type exit to quit): ")

    if query.lower() == "exit":
        break

    query_embedding = model.encode([query])

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        k=3
    )

    if distances[0][0] > 1.5:
    print(
        "\nI could not find this information in the uploaded paper."
    )
    continue

    print("\nTop Relevant Chunks:\n")

    for i in indices[0]:
        print("=" * 80)
        print(chunks[i][:500])
        print()