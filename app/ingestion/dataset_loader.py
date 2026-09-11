import pyarrow.parquet as pq

from app.ingestion.models import Document
from app.ingestion.cleaner import clean_text

def load_documents(file_path: str, limit: int = 100):
    parquet_file = pq.ParquetFile(file_path)

    documents = []
    
    for batch in parquet_file.iter_batches(batch_size=100):

        data = batch.to_pydict()

        for record in range(len(data["query_id"])):

            query_id = data["query_id"][record]

            passages = data["passages"][record]

            english_passages = passages["English_passages"]
            hindi_passages = passages["Translated_passages"]
            selected = passages["is_selected"]

            for passage_index in range(len(english_passages)):

                document = Document(
                    query_id=query_id,
                    passage_index=passage_index,
                    english_text=clean_text(
                        english_passages[passage_index]
                    ),
                    hindi_text=clean_text(
                        hindi_passages[passage_index]
                    ),
                    is_selected=selected[passage_index]
                )

                documents.append(document)

                if len(documents) >= limit:
                    return documents

    return documents
