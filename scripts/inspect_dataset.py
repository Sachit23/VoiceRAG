import pyarrow.parquet as pq

train_file_path = "/Users/kartik/.cache/huggingface/hub/datasets--ai4bharat--MSMARCO-XI/snapshots/bf5cdc1f26e581e519018e434db14edd1b77602b/train/hintrain.parquet"

# from datasets import load_dataset

# def main():
#     dataset = load_dataset(
#             "ai4bharat/MSMARCO-XI",
#             data_files = {
#                 "train": "**/hintrain.*",
#                 "validation": "**/hinval.*"
#             }
#         )
#     print(dataset)
    
#     for split in dataset:
#         print(f"\nSplit: {split}")
#         print(f"Number of rows: {len(dataset[split])}")

#         if len(dataset[split]) > 0:
#             print("\nFirst record:")
#             print(dataset[split][0])

# if __name__ == "__main__":
#     main()


# def main():
#     parquet_file = pq.ParquetFile(train_file_path)

#     print("Number of row groups:", parquet_file.num_row_groups)

#     print("\nColumns:")
#     for column in parquet_file.schema.names:
#         print(" -", column)

#     print("\nReading simple columns only...")

#     table = parquet_file.read(
#         columns=[
#             "source_lang",
#             "target_lang",
#             "Answer",
#             "query_id",
#             "query_type",
#             "Eng_Query",
#             "Eng_Answer",
#             "query"
#         ]
#     )

#     print("\nNumber of rows:", table.num_rows)

#     print("\nFirst record:")
#     print(table.slice(0, 1).to_pylist()[0])


# if __name__ == "__main__":
#     main()

# def main():
#     parquet_file = pq.ParquetFile(train_file_path)

#     print("Columns:")
#     for i, column in enumerate(parquet_file.schema.names):
#         print(i, column)

#     print("\nReading passages column...")

#     passages_table = parquet_file.read(
#         columns=["passages"]
#     )

#     print(passages_table)


# if __name__ == "__main__":
#     main()

def main():
    parquet_file = pq.ParquetFile(train_file_path)

    batch_iter = parquet_file.iter_batches(batch_size=5)
    first_batch = next(batch_iter)
    
    data = first_batch.to_pydict()
    
    for i in range(5):
        print("--- FIRST RECORD PASSAGES ---")
        print("Passages Data Structure:")
        print(data["passages"][i])

if __name__ == "__main__":
    main()

