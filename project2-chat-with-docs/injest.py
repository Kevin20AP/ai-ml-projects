from pypdf import PdfReader

PDF_PATH = "attention.pdf"
SOURCE = "attention.pdf"

def load_pages(pdf_path):
    """Read the PDF and return a list of (page_number, text) tuples."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append((i + 1, text))   # i+1 so pages start at 1, not 0
    return pages

def chunk_text(text, chunk_size=800, overlap=100):
    """Split one page's text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap   # step back by 'overlap' so chunks share a boundary
    return chunks

def build_chunks(pdf_path, source):
    """Turn the whole PDF into a list of chunk records with page metadata."""
    records = []
    for page_num, page_text in load_pages(pdf_path):
        if not page_text.strip():
            continue   # skip blank pages
        for c_index, chunk in enumerate(chunk_text(page_text)):
            records.append({
                "text": chunk,
                "page": page_num,
                "source": source,
                "chunk_id": f"{source}-p{page_num}-c{c_index}",
            })
    return records

if __name__ == "__main__":
    chunks = build_chunks(PDF_PATH, SOURCE)
    print(f"Total chunks: {len(chunks)}\n")
    for rec in chunks[:3]:
        print(f"--- {rec['chunk_id']} (page {rec['page']}) ---")
        print(rec["text"][:300])
        print()