import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.chunking.hybrid import hybrid_chunks


english_text = (
    "The Manhattan Project was a research project during World War II. "
    "It resulted in the development of the first nuclear weapons. "
    "Its legacy continues to affect history and science."
)

hindi_text = (
    "मैनहट्टन परियोजना द्वितीय विश्व युद्ध के दौरान एक शोध परियोजना थी। "
    "इसके परिणामस्वरूप पहले परमाणु हथियारों का विकास हुआ। "
    "इसकी विरासत इतिहास और विज्ञान को प्रभावित करती है।"
)

# Text with an oversized sentence exceeding chunk_size to demonstrate fixed-size fallback
oversized_sentence_text = (
    "Short introductory sentence. "
    "This is an intentionally very long sentence designed to exceed the configured chunk size limit so that we can verify that the hybrid chunking strategy correctly falls back to fixed-size chunking instead of overflowing. "
    "Short concluding sentence."
)


print("\n==============================")
print("ENGLISH CHUNKS")
print("==============================")

english_chunks = hybrid_chunks(
    text=english_text,
    chunk_size=100,
)

print(f"Total chunks: {len(english_chunks)}")
for index, chunk in enumerate(english_chunks):
    print(f"\nChunk {index} ({len(chunk)} chars):")
    print(chunk)


print("\n==============================")
print("HINDI CHUNKS")
print("==============================")

hindi_chunks = hybrid_chunks(
    text=hindi_text,
    chunk_size=100,
)

print(f"Total chunks: {len(hindi_chunks)}")
for index, chunk in enumerate(hindi_chunks):
    print(f"\nChunk {index} ({len(chunk)} chars):")
    print(chunk)


print("\n==============================")
print("OVERSIZED SENTENCE (HYBRID FALLBACK)")
print("==============================")

oversized_chunks = hybrid_chunks(
    text=oversized_sentence_text,
    chunk_size=80,
)

print(f"Total chunks: {len(oversized_chunks)}")
for index, chunk in enumerate(oversized_chunks):
    print(f"\nChunk {index} ({len(chunk)} chars):")
    print(chunk)

