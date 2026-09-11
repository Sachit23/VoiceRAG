from dataclasses import dataclass

@dataclass
class Chunk:
    query_id: int
    passage_index: int
    language: str
    chunk_index: int
    text: str
    chunking_strategy: str
    is_selected: int