from app.chunking.sentence import split_sentences
from app.chunking.fixed_size import fixed_size_chunks


def hybrid_chunks(
    text: str,
    chunk_size: int,
) -> list[str]:

    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    sentences = split_sentences(text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:

        # If the sentence itself is bigger than the limit,
        # split it using fixed-size chunking.
        if len(sentence) > chunk_size:

            # Save whatever we have accumulated before this sentence.
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""

            # Fallback to fixed-size splitting
            oversized_chunks = fixed_size_chunks(
                text=sentence,
                chunk_size=chunk_size,
                overlap=0,
            )

            chunks.extend(oversized_chunks)
            continue

        # Try adding the sentence to the current chunk.
        if not current_chunk:
            current_chunk = sentence
        else:
            candidate = current_chunk + " " + sentence

            if len(candidate) <= chunk_size:
                current_chunk = candidate
            else:
                chunks.append(current_chunk)
                current_chunk = sentence

    # Add the final chunk.
    if current_chunk:
        chunks.append(current_chunk)

    return chunks