import os

from llama_index.core import (StorageContext, VectorStoreIndex,
                              load_index_from_storage)
from llama_index.readers.file import PDFReader


def build_index(data, index_name):
    """
    Build a new index from the provided data and save it to the specified index name.
    """
    print(f"Building index {index_name}")
    index = VectorStoreIndex.from_documents(data, show_progress=True)
    index.storage_context.persist(persist_dir=index_name)
    return index


def load_existing_index(index_name):
    """
    Load an existing index from storage.
    """
    return load_index_from_storage(StorageContext.from_defaults(persist_dir=index_name))


def get_index(data, index_name):
    """
    Retrieve the index for the given data. Build a new index if it doesn't exist.
    """
    if not os.path.exists(index_name):
        index = build_index(data, index_name)
    else:
        index = load_existing_index(index_name)
    return index


def create_pdf_engine(pdf_path):
    """
    Create a query engine for the given PDF file.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        QueryEngine: The query engine for the PDF file.
    """
    pdf_data = PDFReader().load_data(file=pdf_path)
    index_name = "pdf_index"
    pdf_index = get_index(pdf_data, index_name)
    return pdf_index.as_query_engine()
