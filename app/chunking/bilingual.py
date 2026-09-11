from app.ingestion.models import Document
from app.chunking.models import Chunk
from app.chunking.fixed_size import fixed_size_chunks

def create_fixed_size_chunks(
    document: Document,
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    
    chunks = []
    
    languages = {
        "en": document.english_text,
        "hi": document.hindi_text,
    }
    
    for language, text in languages.items():
        
        text_chunks = fixed_size_chunks(
            text = text,
            chunk_size = chunk_size,
            overlap = overlap,
        )
        
        for chunk_index, chunk_text in enumerate(text_chunks):
            chunk = Chunk(
                query_id=document.query_id,
                passage_index=document.passage_index,
                language=language,
                chunk_index=chunk_index,
                text=chunk_text,
                chunking_strategy="fixed_size",
                is_selected=document.is_selected
            )
            chunks.append(chunk)

    return chunks
