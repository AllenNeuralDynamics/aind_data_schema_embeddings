"""Embedding data scehma repository into DocDB"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

from sentence_transformers import SentenceTransformer

from aind_data_schema_embeddings.code_chunker import PythonCodeChunker
from aind_data_schema_embeddings.doc_chunker import DocumentChunker
from aind_data_schema_embeddings.utils import ResourceManager

# folder_path = (
#     r"C:\Users\sreya.kumar\Documents\GitHub\"
#     r"aind_data_schema_embeddings\src\aind_data_schema_embeddings"
# )
# os.chdir(folder_path)

# # Now add this directory to path
# sys.path.append(os.getcwd())


data_schema_src_path = Path(r"C:\Users\sreya.kumar\aind-data-schema-dev\src")
data_schema_read_the_docs_path = Path(
    r"c:\Users\sreya.kumar\Downloads\aind_data_schema_read_the_docs"
)
file_dir = [data_schema_src_path, data_schema_read_the_docs_path]

db_name = "metadata_vector_index"
collection = "aind_data_schema_vectors"
index_name = "vector_embeddings_index"
# vectors stored in vector_embeddings

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename=f"logs/log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="w",
)


model = SentenceTransformer(
    "mixedbread-ai/mxbai-embed-large-v1", truncate_dim=1024
)


def class_to_text(class_instance):
    """Transforms class object to python dictionary"""

    return json.dumps(class_instance.__dict__)


def generate_embeddings_for_batch(batch: list) -> dict:
    """Generates embeddings vectors for a batch of loaded documents"""
    docs_embeddings = model.encode(batch)

    vectors_to_mongodb = [vector.tolist() for vector in docs_embeddings]
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


def chunk_maker(file_name: str, file_path: str):
    """Creating chunks based on file type"""

    if ".py" in file_name:
        logging.info("Code Chunker initialized")
        code_chunker = PythonCodeChunker(
            file_path=str(file_path), file_name=file_name
        )
        logging.info("Creating chunks...")
        chunks = code_chunker.create_chunks()

    if ".txt" in file_name:
        logging.info("Document Chunker initialized")
        doc_chunker = DocumentChunker(
            file_path=str(file_path), file_name=file_name
        )
        logging.info("Creating chunks...")
        chunks = doc_chunker.create_chunks()
        chunks = [class_to_text(chunk) for chunk in chunks]

    # text_chunks = [class_to_text(chunk) for chunk in chunks]
    return chunks


with ResourceManager() as RM:

    collection = RM.client[db_name][collection]

    logging.info("Finding files that have already been embedded")
    embedded_files = set(
        [asset["file_name"] for asset in collection.find({}, {"file_name": 1})]
    )
    logging.info(f"Files already embedded: {embedded_files}")

    logging.info("Going through directory")
    for file_path in (f for path in file_dir for f in path.rglob("*")):
        if file_path.is_file() and file_path.name not in embedded_files:

            logging.info(f"Processing file: {file_path}")
            file_name = file_path.name

            try:
                chunks = chunk_maker(file_name, file_path)
                logging.info("Vectorizing chunks")
                text_and_vector_list = generate_embeddings_for_batch(chunks)
                logging.info("Adding to vectorstore...")
                write_embeddings_to_docdb_for_batch(
                    file_name, collection, text_and_vector_list
                )

            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")
