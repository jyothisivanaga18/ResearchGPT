from pypdf import PdfReader
from pathlib import Path

pdf_path = Path("data/attention.pdf")

reader = PdfReader(pdf_path)

print(f"Total Pages: {len(reader.pages)}")

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text

print("\nFirst 1000 Characters:\n")
print(text[:1000])