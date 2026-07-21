import pdfplumber
import io  # Import the io module

def extract_text_from_pdf(file: bytes) -> str:
    text = ""
    # Wrap the raw bytes in a BytesIO object so pdfplumber can 'seek' through it
    with pdfplumber.open(io.BytesIO(file)) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text.strip()

def chunk_text(text: str, max_length: int = 500) -> list[str]:
    """Splits text into chunks for better vector embedding."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) < max_length:
            current_chunk.append(word)
            current_length += len(word) + 1
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks