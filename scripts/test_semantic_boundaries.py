import re

from sentence_transformers import util
from app.embeddings.embedding_service import EmbeddingService


def main():
    # Deliberately contains two topics: nuclear research and mango farming.
    # text = (
    #     "The Manhattan Project developed the first atomic bombs. "
    #     "Scientists worked in secret laboratories to study nuclear reactions. "
    #     "Their research led to the creation of nuclear weapons. "
    #     "Mango trees grow best in warm tropical climates. "
    #     "Farmers water the trees and protect them from pests. "
    #     "Ripe mangoes are harvested and sold in fruit markets."
    # )
    # text = (
    #     "Mango trees grow best in warm tropical climates. "
    #     "The trees need sunlight and regular watering to grow well. "
    #     "Farmers protect mango trees from insects and diseases. "
    #     "The fruit is harvested when it becomes ripe."
    # )
    # text = (
    #     "आम के पेड़ गर्म जलवायु में अच्छी तरह बढ़ते हैं। "
    #     "इन पेड़ों को बढ़ने के लिए धूप और नियमित पानी की जरूरत होती है। "
    #     "किसान आम के पेड़ों को कीड़ों और बीमारियों से बचाते हैं। "
    #     "पकने पर आम के फलों की कटाई की जाती है।"
    # )
    text = (
        "मैनहट्टन परियोजना ने पहले परमाणु बम विकसित किए। "
        "वैज्ञानिकों ने गुप्त प्रयोगशालाओं में परमाणु प्रतिक्रियाओं का अध्ययन किया। "
        "उनके शोध से परमाणु हथियारों का निर्माण हुआ। "
        "आम के पेड़ गर्म जलवायु में अच्छी तरह बढ़ते हैं। "
        "किसान पेड़ों को पानी देते हैं और कीड़ों से बचाते हैं। "
        "पके आमों को तोड़कर फलों के बाजार में बेचा जाता है।"
    )

    # Simple splitter for this controlled example.
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?।])\s+", text)
        if sentence.strip()
    ]

    if len(sentences) < 2:
        raise SystemExit("At least two sentences are needed.")

    print("Sentences:")

    for number, sentence in enumerate(sentences, start=1):
        print(f"{number}. {sentence}")

    embedding_service = EmbeddingService()
    embeddings = embedding_service.embed_sentences(sentences)

    print("\nEmbedding shape:", embeddings.shape)
    print("\nNeighboring sentence similarities:")

    similarities = []

    for index in range(len(sentences) - 1):
        score = util.cos_sim(
            embeddings[index],
            embeddings[index + 1],
        ).item()

        similarities.append(score)

        print(
            f"Sentence {index + 1} → Sentence {index + 2}: "
            f"{score:.4f}"
        )

    weakest_index = min(
        range(len(similarities)),
        key=lambda index: similarities[index],
    )

    print("\nLeast similar neighboring pair:")
    print("Before:", sentences[weakest_index])
    print("After: ", sentences[weakest_index + 1])
    print("Candidate boundary—not an automatic split yet.")
    
    similarity_threshold = 0.80
    chunks = []
    current_sentences = [sentences[0]]

    for index, score in enumerate(similarities):
        if score < similarity_threshold:
            chunks.append(" ".join(current_sentences))
            current_sentences = []

        current_sentences.append(sentences[index + 1])

    chunks.append(" ".join(current_sentences))

    print(f"\nCreated {len(chunks)} chunks:")

    for number, chunk in enumerate(chunks, start=1):
        print(f"\nChunk {number}:")
        print(chunk)


if __name__ == "__main__":
    main()