import pickle

with open("vectorstore/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print("Total Chunks:", len(chunks))

print("\nFIRST CHUNK:\n")
print(chunks[0][:3000])