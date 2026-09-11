import re

def split_sentences(text: str) -> list[str]:
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?।])\s+", text)

    return [sentence.strip() for sentence in sentences if sentence.strip()]

def sentence_chunks(
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

        if not current_chunk:
            current_chunk = sentence
            continue

        candidate = current_chunk + " " + sentence

        if len(candidate) <= chunk_size:
            current_chunk = candidate
        else:
            chunks.append(current_chunk)
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def sentence_chunks_with_overlap(
    text: str,
    chunk_size: int,
    overlap_sentences: int = 1,
) -> list[str]:

    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap_sentences < 0:
        raise ValueError("overlap_sentences cannot be negative")

    sentences = split_sentences(text)

    chunks = []
    current_sentences = []

    for sentence in sentences:

        candidate_sentences = current_sentences + [sentence]
        candidate = " ".join(candidate_sentences)

        if current_sentences and len(candidate) > chunk_size:
            chunks.append(" ".join(current_sentences))

            overlap = current_sentences[-overlap_sentences:] \
                if overlap_sentences > 0 else []

            current_sentences = overlap + [sentence]

        else:
            current_sentences = candidate_sentences

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks
