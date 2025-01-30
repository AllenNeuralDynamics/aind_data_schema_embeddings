"""Embedding data scehma repository into DocDB"""

import json
import logging
from pathlib import Path

from code_chunker import PythonCodeChunker
from sentence_transformers import SentenceTransformer
from utils import ResourceManager

root_path = Path(r"C:\Users\sreya.kumar\aind-data-schema-dev\src")

db_name = "metadata_vector_index"
collection = "aind_data_schema_vectors"
index_name = "vector_embeddings_index"
# vectors stored in vector_embeddings

logging.basicConfig(
    filename="vector_store.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="w",
)

model = SentenceTransformer(
    "dunzhang/stella_en_1.5B_v5", device="cpu", trust_remote_code=True
)


def class_to_text(class_instance):
    """Transforms class object to python dictionary"""
    return json.dumps(class_instance.__dict__)


def generate_embeddings_for_batch(batch: list) -> dict:
    """Generates embeddings vectors for a batch of loaded documents"""
    schema_embeddings = model.encode(batch, batch_size=len(batch))
    vectors_to_mongodb = [vector.tolist() for vector in schema_embeddings]

    text_and_vector_list = list(zip(batch, vectors_to_mongodb))

    return text_and_vector_list


def write_embeddings_to_docdb_for_batch(
    file_name: str, collection, text_and_vector_list: list
) -> None:
    """Writes vectors into DocDB in batches"""

    for text, vector in text_and_vector_list:
        document = {
            "file_name": file_name,
            "text": text,
            "vector_embeddings": vector,
        }
        try:
            result = collection.insert_one(document)
            logging.info(f"Inserted document with ID: {result.inserted_id}")
        except Exception as e:
            logging.error(f"Error inserting document: {e}")


with ResourceManager() as RM:

    collection = RM.client[db_name][collection]

    logging.info("Going through directory")
    for file_path in root_path.rglob("*"):
        if file_path.is_file():
            logging.info(f"Processing file: {file_path}")
            try:
                file_name = file_path.name
                logging.info("Chunker initialized")
                chunker = PythonCodeChunker(str(file_path))

                logging.info("Creating chunks...")
                chunks = chunker.create_chunks()
                text_chunks = [class_to_text(chunk) for chunk in chunks]

                logging.info("Vectorizing chunks")
                text_and_vector_list = generate_embeddings_for_batch(
                    text_chunks
                )

                logging.info("Adding to vectorstore...")
                write_embeddings_to_docdb_for_batch(
                    file_name, collection, text_and_vector_list
                )

            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")
