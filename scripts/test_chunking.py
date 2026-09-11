import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.chunking.fixed_size import fixed_size_chunks

text = (
    "The Manhattan Project and its atomic bomb helped bring an end to "
    "World War II. Its legacy of peaceful uses of atomic energy continues "
    "to have an impact on history and science."
)

chunks = fixed_size_chunks(
    text=text,
    chunk_size=100,
    overlap=20,
)

print(f"Total chunks: {len(chunks)}")

for index, chunk in enumerate(chunks):
    print("\n---")
    print(f"Chunk {index}")
    print(chunk)
