import sys

from dataclasses import asdict
from uuid import NAMESPACE_URL, uuid5
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from app.chunking.bilingual import create_fixed_size_chunks
from app.embeddings.embedding_service import EmbeddingService
from app.ingestion.dataset_loader import load_documents


COLLECTION_NAME = "voicerag_e5small_fixed400_overlap50_v1"


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python -m scripts.build_index <hintrain_parquet_path>"
        )

    # 1. Load the same small training sample.
    documents = load_documents(
        file_path=sys.argv[1],
        limit=100,
    )

    chunks = []

    for document in documents:
        chunks.extend(
            chunk
            for chunk in create_fixed_size_chunks(
                document=document,
                chunk_size=400,
                overlap=50,
            )
            if chunk.text.strip()
        )

    if not chunks:
        raise SystemExit("No non-empty chunks were produced.")

    print("Passage pairs loaded:", len(documents))
    print("Chunks created:", len(chunks))

    # 2. Generate vectors using our existing service.
    embedding_service = EmbeddingService()
    embeddings = embedding_service.embed_chunks(chunks)

    vector_size = embeddings.shape[1]
    print("Embedding matrix shape:", embeddings.shape)

    client = QdrantClient(
        url="http://localhost:6333",
        timeout=30,
    )

    try:
        # 3. Create the collection only if it does not exist.
        if not client.collection_exists(COLLECTION_NAME):
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )
            print("Created collection:", COLLECTION_NAME)
        else:
            config = client.get_collection(
                COLLECTION_NAME
            ).config.params.vectors

            if (
                not isinstance(config, VectorParams)
                or config.size != vector_size
                or config.distance != Distance.COSINE
            ):
                raise ValueError(
                    "Existing collection has incompatible vector settings."
                )

        # 4. Associate each vector with its original chunk.
        points = []

        for chunk, vector in zip(chunks, embeddings):
            identity = (
                f"MSMARCO-XI/hin/train/{COLLECTION_NAME}/"
                f"{chunk.query_id}/{chunk.passage_index}/"
                f"{chunk.language}/{chunk.chunk_index}"
            )

            point = PointStruct(
                id=str(uuid5(NAMESPACE_URL, identity)),
                vector=vector.tolist(),
                payload={
                    **asdict(chunk),
                    "dataset": "ai4bharat/MSMARCO-XI",
                    "split": "train",
                    "embedding_model": "intfloat/multilingual-e5-small",
                },
            )

            points.append(point)

        # 5. Save points in small batches.
        batch_size = 64

        for start in range(0, len(points), batch_size):
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points[start:start + batch_size],
                wait=True,
            )

        count = client.count(
            collection_name=COLLECTION_NAME,
            exact=True,
        ).count

        print("\nIndexing complete.")
        print("Collection:", COLLECTION_NAME)
        print("Stored points:", count)

    finally:
        client.close()


if __name__ == "__main__":
    main()
