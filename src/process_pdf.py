from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle


def process_pdf(pdf_path):

    print("Reading PDF...")

    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text

    print("Chunking...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)

    print(f"Total Chunks: {len(chunks)}")

    print("Creating Embeddings...")

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    embeddings = model.encode(chunks)

    embeddings = np.array(
        embeddings
    ).astype("float32")

    print("Building FAISS Index...")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(embeddings)

    print("Saving FAISS Index...")

    faiss.write_index(
        index,
        "vectorstore/index.faiss"
    )

    print("Saving Chunks...")

    with open(
        "vectorstore/chunks.pkl",
        "wb"
    ) as f:

        pickle.dump(
            chunks,
            f
        )

    return len(chunks)