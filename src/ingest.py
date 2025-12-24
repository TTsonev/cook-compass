import os
import logging
from pathlib import Path
from haystack import Pipeline
from haystack.components.converters import CSVToDocument
from haystack.components.embedders import HuggingFaceAPIDocumentEmbedder
from haystack.components.preprocessors import CSVDocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.utils import Secret
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

from .utils import PROJECT_ROOT, load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_embedder(model_name, api_token):
    return HuggingFaceAPIDocumentEmbedder(
        api_type="serverless_inference_api",
        api_params={"model": model_name},
        token=Secret.from_token(api_token) if api_token else None
    )

def create_indexing_pipeline(document_store, embedder):
    indexing_pipe = Pipeline()
    indexing_pipe.add_component("converter", CSVToDocument())
    indexing_pipe.add_component("splitter", CSVDocumentSplitter(split_mode='row-wise'))
    indexing_pipe.add_component("embedder", embedder)
    indexing_pipe.add_component("writer", DocumentWriter(document_store))
    indexing_pipe.connect("converter", "splitter")
    indexing_pipe.connect("splitter", "embedder")
    indexing_pipe.connect("embedder", "writer")
    return indexing_pipe

def ingest_csv_data():
    api_token = os.getenv("API_TOKEN")

    config = load_config()

    paths = config["paths"]
    chroma_path = PROJECT_ROOT / paths["chroma"]
    data_path = PROJECT_ROOT / paths["data"]
    
    vector_store = ChromaDocumentStore(persist_path=chroma_path)

    logger.info("Starting data ingestion from %s", data_path)
    
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Error: Data path not found at {data_path}")
    file_paths = list(data_path.glob("*.csv"))

    document_embedder = get_embedder(config['embeddings_model'], api_token)
    indexing_pipe = create_indexing_pipeline(vector_store, document_embedder)
    
    indexing_pipe.run({"converter": {"sources": file_paths}})
    logger.info("Data ingestion completed.")

if __name__ == "__main__":
    ingest_csv_data()
