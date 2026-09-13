from qdrant_client import QdrantClient

from app.embeddings.embedding_service import EmbeddingService


COLLECTION_NAME = "voicerag_e5small_fixed400_overlap50_v1"


def main():
    client = QdrantClient(
        url="http://localhost:6333",
        timeout=30,
    )

    try:
        if not client.collection_exists(COLLECTION_NAME):
            raise SystemExit(
                "Collection not found. Run scripts.build_index first."
            )

        count = client.count(
            collection_name=COLLECTION_NAME,
            exact=True,
        ).count

        if count == 0:
            raise SystemExit("The collection is empty. Index some chunks first.")

        print("Stored chunks:", count)

        # Load once and reuse across questions.
        embedding_service = EmbeddingService()

        print("\nEnter an English or Hindi question.")
        print("Type 'exit' to stop.")

        while True:
            query = input("\nQuestion: ").strip()

            if query.lower() == "exit":
                break

            if not query:
                print("Please enter a question.")
                continue

            try:
                query_embedding = embedding_service.embed_query(query)
            except ValueError as error:
                print(error)
                continue

            # Qdrant compares the query vector with stored chunk vectors.
            results = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_embedding.tolist(),
                limit=5,
                with_payload=True,
                with_vectors=False,
            ).points

            if not results:
                print("No results returned.")
                continue

            for rank, result in enumerate(results, start=1):
                payload = result.payload or {}

                print(f"\nRank {rank} | Similarity: {result.score:.4f}")
                print(
                    f"Language: {payload.get('language')} | "
                    f"Query ID: {payload.get('query_id')} | "
                    f"Passage: {payload.get('passage_index')} | "
                    f"Chunk: {payload.get('chunk_index')}"
                )
                print(payload.get("text", ""))

    finally:
        client.close()


if __name__ == "__main__":
    main()