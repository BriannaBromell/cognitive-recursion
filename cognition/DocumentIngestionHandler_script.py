import os
import argparse
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions


import os
from collections import deque
from typing import Deque
import asyncio

import aiofiles

async def read_in_chunks(file_path: str, encodings, chunk_size, overlap):
    for encoding in encodings:
        try:
            queue = deque(maxlen=chunk_size)
            async with aiofiles.open(file_path, mode='r', encoding=encoding) as f:
                async for line in f:
                    queue.append(line)
                    if len(queue) == chunk_size:
                        yield ' '.join(list(queue))
                        for _ in range(overlap):
                            queue.popleft()
            break
        except UnicodeDecodeError:
            continue

async def main(
    collection_name: str = "documents_collection",
    documents_directory: str = "documents",
    persist_directory: str = ".",
) -> None:
    chunk_size = 5
    overlap = 2 
    documents = []
    metadatas = []
    encodings = ["utf-8", "latin-1", "utf-16", "cp1252"]
    files = os.listdir(documents_directory)

    for filename in files:
        file_path = os.path.join(documents_directory, filename)
        async for chunk in read_in_chunks(file_path, encodings, chunk_size, overlap):
            documents.append(chunk)
            metadatas.append({"filename": filename, "chunk": chunk})

    embeddings = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection(name=collection_name, embedding_function=embeddings)

    print(f"Collection already contains {count} documents")


    count = collection.count()
    ids = [str(i) for i in range(count, count + len(documents))]
    for i in range(0, len(documents), 100):
        collection.add(
            ids=ids[i : i + 100],
            documents=documents[i : i + 100],
            metadatas=metadatas[i : i + 100],  # type: ignore
        )
    new_count = collection.count()
    print(f"Added {new_count - count} documents")

# Use asyncio.run(main()) when calling the function
            
'''
def main(
    collection_name: str = "documents_collection",
    documents_directory: str = "documents",
    persist_directory: str = ".",
) -> None:
    # Read all files in the data directory
    documents = []
    metadatas = []
    encodings = ["utf-8", "latin-1", "utf-16", "cp1252"]
    files = os.listdir(documents_directory)
    for filename in files:
        encoding_error = True
        for encoding in encodings:
            try:
                with open(f"{documents_directory}/{filename}", "r", encoding=encoding) as file:
                    for line_number, line in enumerate(tqdm(file.readlines(), desc=f"Reading {filename}"), 1):
                        line = line.strip()                # Strip whitespace and append the line to the documents list
                        documents.append(line)
                        metadatas.append({"filename": filename, "line_number": line_number})
                    encoding_error = False
                    break
            except UnicodeDecodeError:
                continue
        if encoding_error:
            try:
                with open(f"{documents_directory}/{filename}", "r", errors="ignore") as file:
                    for line_number, line in enumerate(tqdm(file.readlines(), desc=f"Reading {filename}"), 1):
                        line = line.strip()
                        documents.append(line)
                        metadatas.append({"filename": filename, "line_number": line_number})
            except UnicodeDecodeError:
                print(f"Failed to decode file: {filename}")
            

    # Continue with your code after successfully processing the files

    # Instantiate a persistent chroma client in the persist_directory.
    # Learn more at docs.trychroma.com
    embeddings = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path=persist_directory)

    # If the collection already exists, we just return it. This allows us to add more
    # data to an existing collection.
    collection = client.get_or_create_collection(name=collection_name, embedding_function=embeddings)

    # Create ids from the current count
    count = collection.count()
    print(f"Collection already contains {count} documents")
    ids = [str(i) for i in range(count, count + len(documents))]

    # Load the documents in batches of 100    
    for i in tqdm(
        range(0, len(documents), 100), desc="Adding documents", unit_scale=100
    ):
        collection.add(
            ids=ids[i : i + 100],
            documents=documents[i : i + 100],
            metadatas=metadatas[i : i + 100],  # type: ignore
        )

    new_count = collection.count()
    print(f"Added {new_count - count} documents")
'''

if __name__ == "__main__":
    # Read the data directory, collection name, and persist directory
    parser = argparse.ArgumentParser(
        description="Load documents from a directory into a Chroma collection"
    )

    # Add arguments
    parser.add_argument(
        "--data_directory",
        type=str,
        default="documents",
        help="The directory where your text files are stored",
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="documents_collection",
        help="The name of the Chroma collection",
    )
    parser.add_argument(
        "--persist_directory",
        type=str,
        default="chroma_storage",
        help="The directory where you want to store the Chroma collection",
    )

    # Parse arguments
    args = parser.parse_args()
    # Run the asynchronous main function
    asyncio.run(main(
        documents_directory=args.data_directory,
        collection_name=args.collection_name,
        persist_directory=args.persist_directory,
    ))

    '''
    main(
        documents_directory=args.data_directory,
        collection_name=args.collection_name,
        persist_directory=args.persist_directory,
    )'''
