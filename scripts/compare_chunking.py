import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ingestion.dataset_loader import load_documents
from app.chunking.fixed_size import fixed_size_chunks
from app.chunking.sentence import sentence_chunks
from app.chunking.sentence import sentence_chunks_with_overlap

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

# print("\n==============================")
# print("ENGLISH - FIXED SIZE")
# print("==============================")

# fixed_en = fixed_size_chunks(
#     text=document.english_text,
#     chunk_size=100,
#     overlap=20,
# )

# for index, chunk in enumerate(fixed_en):
#     print(f"\nChunk {index} ({len(chunk)} chars)")
#     print(chunk)


print("\n==============================")
print("ENGLISH - SENTENCE")
print("==============================")

# sentence_en = sentence_chunks(
#     text=document.english_text,
#     chunk_size=100,
# )

english_chunks = sentence_chunks_with_overlap(
    text=document.english_text,
    chunk_size=100,
    overlap_sentences=1,
)

for index, chunk in enumerate(english_chunks):
    print(f"\nChunk {index} ({len(chunk)} chars)")
    print(chunk)


# print("\n==============================")
# print("HINDI - FIXED SIZE")
# print("==============================")

# fixed_hi = fixed_size_chunks(
#     text=document.hindi_text,
#     chunk_size=100,
#     overlap=20,
# )

# for index, chunk in enumerate(fixed_hi):
#     print(f"\nChunk {index} ({len(chunk)} chars)")
#     print(chunk)


print("\n==============================")
print("HINDI - SENTENCE")
print("==============================")

hindi_chunks = sentence_chunks_with_overlap(
    text=document.hindi_text,
    chunk_size=100,
    overlap_sentences=1,
)

for index, chunk in enumerate(hindi_chunks):
    print(f"\nChunk {index} ({len(chunk)} chars)")
    print(chunk)
