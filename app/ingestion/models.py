from dataclasses import dataclass
   
@dataclass
class Document:
    query_id: int
    passage_index: int
    english_text: str
    hindi_text: str
    is_selected: int