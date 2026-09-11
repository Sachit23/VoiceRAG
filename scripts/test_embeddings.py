from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("intfloat/multilingual-e5-small")

chunks = [
    "The Manhattan Project developed atomic weapons.",
    "मैनहट्टन परियोजना ने परमाणु हथियार विकसित किए।",
    "Mangoes grow in warm climates.",
]

query = "Which project created the atomic bomb?"
#query = "किस परियोजना ने परमाणु बम बनाया?"

passage_inputs = ["passage: " + chunk for chunk in chunks]
query_input = "query: " + query

chunk_embeddings = model.encode(
    passage_inputs,
    normalize_embeddings=True,
)

query_embedding = model.encode(
    query_input,
    normalize_embeddings=True,
)

print("Chunk embeddings shape:", chunk_embeddings.shape)
print("Query embedding shape:", query_embedding.shape)
print("First chunk's first 10 values:", chunk_embeddings[0][:10])

scores = util.cos_sim(query_embedding, chunk_embeddings)[0]

ranked_results = sorted(
    zip(chunks, scores.tolist()),
    key=lambda item: item[1],
    reverse=True,
)

print("\nQuery:", query)
print("\nRanked chunks:")

for rank, (text, score) in enumerate(ranked_results, start=1):
    print(f"{rank}. Similarity: {score:.4f}")
    print(f"   {text}")

