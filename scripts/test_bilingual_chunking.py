import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ingestion.dataset_loader import load_documents
from app.chunking.bilingual import create_fixed_size_chunks


file_path = (
    "/Users/kartik/.cache/huggingface/hub/"
    "datasets--ai4bharat--MSMARCO-XI/snapshots/"
    "bf5cdc1f26e581e519018e434db14edd1b77602b/"
    "train/hintrain.parquet"
)

documents = load_documents(
    file_path=file_path,
    limit=1,
)

document = documents[0]

chunks = create_fixed_size_chunks(
    document=document,
    chunk_size=100,
    overlap=20,
)

print(f"Total chunks: {len(chunks)}")

for chunk in chunks:
    print("\n----------------")
    print(f"Query ID: {chunk.query_id}")
    print(f"Passage Index: {chunk.passage_index}")
    print(f"Language: {chunk.language}")
    print(f"Chunk Index: {chunk.chunk_index}")
    print(f"Strategy: {chunk.chunking_strategy}")
    print(f"Selected: {chunk.is_selected}")
    print(f"Text: {chunk.text}")
