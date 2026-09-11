import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.chunking.sentence import split_sentences
from app.chunking.sentence import sentence_chunks


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


print("\n\nENGLISH CHUNKS")
print("--------------")

english_chunks = sentence_chunks(
    text=english_text,
    chunk_size=100,
)

for index, chunk in enumerate(english_chunks):
    print(f"\nChunk {index} ({len(chunk)} chars):")
    print(chunk)


print("\n\nHINDI CHUNKS")
print("------------")

hindi_chunks = sentence_chunks(
    text=hindi_text,
    chunk_size=100,
)

for index, chunk in enumerate(hindi_chunks):
    print(f"\nChunk {index} ({len(chunk)} chars):")
    print(chunk)
