import sys

from sentence_transformers import util

from app.ingestion.dataset_loader import load_documents
from app.chunking.bilingual import create_fixed_size_chunks
from app.embeddings.embedding_service import EmbeddingService


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python -m scripts.test_dataset_embeddings <parquet_path>"
        )

    # 1. Load a small sample of real passage pairs.
    documents = load_documents(
        file_path=sys.argv[1],
        limit=100,
    )

    # 2. Create English and Hindi chunks.
    chunks = []

    for document in documents:
        passage_chunks = create_fixed_size_chunks(
            document=document,
            chunk_size=400,
            overlap=50,
        )

        chunks.extend(
            chunk for chunk in passage_chunks if chunk.text.strip()
        )

    if not chunks:
        raise SystemExit("No non-empty chunks were produced.")

    print("Passage pairs loaded:", len(documents))
    print("Chunks created:", len(chunks))

    print("\nSample chunks:")

    for chunk in chunks[:4]:
        print(f"\nLanguage: {chunk.language}")
        print(chunk.text)

    # 3. Load the model once through the reusable service.
    embedding_service = EmbeddingService()

    # The service adds prefixes, checks token limits, and normalizes vectors.
    chunk_embeddings = embedding_service.embed_chunks(chunks)

    print("\nEmbedding matrix shape:", chunk_embeddings.shape)

    # 4. Read and embed the question.
    query = input("\nEnter a question about the sample content: ").strip()

    if not query:
        raise SystemExit("Please enter a non-empty question.")

    query_embedding = embedding_service.embed_query(query)

    # 5. Compare the question against every chunk.
    scores = util.cos_sim(query_embedding, chunk_embeddings)[0]

    top_k = min(5, len(chunks))
    top_results = scores.topk(k=top_k)

    print("\nQuery:", query)
    print("\nTop matching chunks:")

    # Each embedding index corresponds to the same index in chunks.
    for rank, (score, index) in enumerate(
        zip(top_results.values.tolist(), top_results.indices.tolist()),
        start=1,
    ):
        chunk = chunks[index]

        print(f"\nRank {rank} | Similarity: {score:.4f}")
        print(
            f"Language: {chunk.language} | "
            f"Query ID: {chunk.query_id} | "
            f"Passage: {chunk.passage_index} | "
            f"Chunk: {chunk.chunk_index}"
        )
        print(chunk.text)


if __name__ == "__main__":
    main()
