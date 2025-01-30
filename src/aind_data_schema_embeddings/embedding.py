"""Embedding data scehma repository into DocDB"""

import json
import logging
from pathlib import Path

from sentence_transformers import SentenceTransformer

from aind_data_schema_embeddings.code_chunker import PythonCodeChunker
from aind_data_schema_embeddings.doc_chunker import DocumentChunker
from aind_data_schema_embeddings.utils import ResourceManager

data_schema_src_path = Path(r"C:\Users\sreya.kumar\aind-data-schema-dev\src")
data_schema_read_the_docs_path = Path(r"c:\Users\sreya.kumar\Downloads\aind_data_schema_read_the_docs")
file_dir = [data_schema_src_path, data_schema_read_the_docs_path]

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

def chunk_maker(file_name: str, file_path:str):
    '''Creating chunks based on file type'''

    if ".py" in file_name: 
        logging.info("Code Chunker initialized")
        code_chunker = PythonCodeChunker(file_path = str(file_path), 
                                    file_name = file_name)
        logging.info("Creating chunks...")
        chunks = code_chunker.create_chunks()
    
    if ".txt" in file_name:
        logging.info("Document Chunker initialized")
        doc_chunker = DocumentChunker(file_path = str(file_path), 
                                  file_name = file_name)
        logging.info("Creating chunks...")
        chunks = doc_chunker.create_chunks()

    text_chunks = [class_to_text(chunk) for chunk in chunks]
    return text_chunks




with ResourceManager() as RM:

    collection = RM.client[db_name][collection]

    logging.info("Finding files that have already been embedded")
    embedded_files = set([asset["file_name"] for asset in 
                                  collection.find({},{ "file_name": 1 })])
    logging.info(f"Files already embedded: {embedded_files}")

    logging.info("Going through directory")
    for file_path in (f for path in file_dir for f in path.rglob("*")):
        if file_path.is_file() and file_path.name not in embedded_files:

            logging.info(f"Processing file: {file_path}")
            file_name = file_path.name
            
            try:
                chunks = chunk_maker(file_name, file_path)
                logging.info("Vectorizing chunks")
                text_and_vector_list = generate_embeddings_for_batch(
                    chunks
                )
                logging.info("Adding to vectorstore...")
                write_embeddings_to_docdb_for_batch(
                    file_name, collection, text_and_vector_list
                )
        
            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")
