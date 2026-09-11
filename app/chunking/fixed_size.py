from app.chunking.models import Chunk

def fixed_size_chunks(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:

    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])

        start = end - overlap

    return chunks