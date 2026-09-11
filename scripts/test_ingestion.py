import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ingestion.dataset_loader import load_documents


TRAIN_FILE_PATH = "/Users/kartik/.cache/huggingface/hub/datasets--ai4bharat--MSMARCO-XI/snapshots/bf5cdc1f26e581e519018e434db14edd1b77602b/train/hintrain.parquet"


def main():
    documents = load_documents(
        TRAIN_FILE_PATH,
        limit=20
    )

    print(f"Loaded documents: {len(documents)}")

    for document in documents[:3]:
        print("\n-------------------------")
        print("Query ID:", document.query_id)
        print("Passage Index:", document.passage_index)
        print("Selected:", document.is_selected)

        print("\nEnglish:")
        print(document.english_text)

        print("\nHindi:")
        print(document.hindi_text)


if __name__ == "__main__":
    main()
