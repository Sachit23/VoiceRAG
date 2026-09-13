import numpy as np

from sentence_transformers import SentenceTransformer
from app.chunking.models import Chunk


class EmbeddingService:
    def __init__(self):
        self.model = SentenceTransformer(
            "intfloat/multilingual-e5-small"
        )

    def _validate_inputs(self, texts: list[str]) -> None:
        tokenized = self.model.tokenizer(
            texts,
            truncation=False,
            padding=False,
            add_special_tokens=True,
        )

        for index, token_ids in enumerate(tokenized["input_ids"]):
            if len(token_ids) > self.model.max_seq_length:
                raise ValueError(
                    f"Input {index} has {len(token_ids)} tokens; "
                    f"maximum is {self.model.max_seq_length}."
                )

    def embed_chunks(self, chunks: list[Chunk]) -> np.ndarray:
        if not chunks:
            raise ValueError("Provide at least one chunk.")

        if any(not chunk.text.strip() for chunk in chunks):
            raise ValueError("Chunk text cannot be empty.")

        texts = ["passage: " + chunk.text for chunk in chunks]

        self._validate_inputs(texts)

        return self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

    def embed_query(self, query: str) -> np.ndarray:
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        text = "query: " + query.strip()

        self._validate_inputs([text])

        return self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
    
    def embed_sentences(self, sentences: list[str]) -> np.ndarray:
        if not sentences:
            raise ValueError("Provide at least one sentence.")

        if any(not sentence.strip() for sentence in sentences):
            raise ValueError("Sentences cannot be empty.")

        texts = ["query: " + sentence.strip() for sentence in sentences]

        self._validate_inputs(texts)

        return self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
    