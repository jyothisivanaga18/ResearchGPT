from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Read PDF
reader = PdfReader("data/attention.pdf")

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text

# Create chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_text(text)

print(f"Total Chunks: {len(chunks)}")

# Load embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Convert first chunk into embedding
embedding = model.encode(chunks[0])

print("\nEmbedding Dimension:")
print(len(embedding))

print("\nFirst 10 Values:")
print(embedding[:10])