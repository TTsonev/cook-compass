import ast
import logging
import os
from pathlib import Path
from typing import List, Any, Tuple

import pandas as pd
from haystack import Pipeline, Document
from haystack.components.embedders import HuggingFaceAPIDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.utils import Secret
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

from .utils import PROJECT_ROOT, load_config

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger("Ingestion")


def _get_embedder(model_name: str, api_token: str) -> HuggingFaceAPIDocumentEmbedder:
    """Initializes the Hugging Face Embedder component."""
    return HuggingFaceAPIDocumentEmbedder(
        api_type="serverless_inference_api",
        api_params={"model": model_name},
        token=Secret.from_token(api_token)
    )


def _clean_list_string(text: Any) -> str:
    """Parses string lists from CSV (e.g. "['a', 'b']") into clean strings ("a, b")."""
    try:
        if isinstance(text, str) and text.strip().startswith("["):
            item_list = ast.literal_eval(text)
            return ", ".join(item_list)
        return str(text)
    except Exception:
        return str(text)

def _load_and_validate_config() -> Tuple[Any, str]:
    """Loads config and ensures API token is present."""
    api_token = os.getenv("API_TOKEN")
    if not api_token:
        logger.critical("API_TOKEN not found in environment. Cannot proceed.")
        raise ValueError("Missing API_TOKEN")

    config = load_config()
    return config, api_token


def _initialize_store(chroma_path: Path) -> ChromaDocumentStore:
    """Sets up the Vector Database connection."""
    try:
        store = ChromaDocumentStore(persist_path=str(chroma_path))
        logger.info(f"ChromaDB initialized at: {chroma_path}")
        return store
    except Exception as e:
        logger.critical(f"Failed to initialize ChromaDB: {e}")
        raise e


def _load_recipe_data(data_dir: Path, filename: str) -> pd.DataFrame:
    """Loads the CSV file specified in the config."""
    file_path = data_dir / filename

    if not file_path.exists():
        logger.critical(f"Data file not found: {file_path}")
        raise FileNotFoundError(f"File {filename} missing in {data_dir}")

    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded '{filename}' with {len(df)} rows.")
        return df
    except Exception as e:
        logger.critical(f"Error reading CSV: {e}")
        raise e


def _transform_data_to_documents(df: pd.DataFrame) -> List[Document]:
    """
    Converts DataFrame rows into Haystack Documents.
    Ingredients -> Content, Steps -> Metadata.
    """
    documents = []
    total_rows = len(df)
    log_interval = max(1, total_rows // 5)

    logger.info("Starting data transformation...")

    for index, row in df.iterrows():
        try:
            clean_ingredients = _clean_list_string(row.get('ingredients', row.get('tags', '')))
            clean_steps = _clean_list_string(row.get('steps', ''))
            clean_tags = _clean_list_string(row.get('tags', ''))

            content_for_embedding = (
                f"Recipe: {row['name']}\n"
                f"Ingredients: {clean_ingredients}\n"
            )

            meta_data = {
                "name": row['name'],
                "steps": clean_steps,
                "minutes": row.get('minutes', 0),
                "original_id": row.get('id', index),
                "tags": clean_tags
            }

            doc = Document(content=content_for_embedding, meta=meta_data)
            documents.append(doc)

            if (index + 1) % log_interval == 0:
                logger.info(f"Processed {index + 1}/{total_rows} recipes.")

        except Exception as e:
            logger.warning(f"Skipping row {index} due to error: {e}")
            continue

    logger.info(f"Transformation complete. {len(documents)} documents created.")
    return documents


def _run_indexing_pipeline(documents: List[Document], store: ChromaDocumentStore, model_name: str, api_token: str):
    """Creates and runs the Haystack pipeline to embed and write documents."""
    try:
        logger.info("Building Indexing Pipeline...")
        pipe = Pipeline()

        embedder = _get_embedder(model_name, api_token)
        writer = DocumentWriter(store)

        pipe.add_component("embedder", embedder)
        pipe.add_component("writer", writer)
        pipe.connect("embedder", "writer")

        logger.info(f"Running Pipeline with model: {model_name}...")
        pipe.run({"embedder": {"documents": documents}})

        logger.info("--- Ingestion Pipeline Finished Successfully ---")

    except Exception as e:
        logger.critical(f"Pipeline execution failed: {e}")
        raise e


def ingest_csv_data():
    config, api_token = _load_and_validate_config()

    paths = config["paths"]
    chroma_path = PROJECT_ROOT / paths["chroma"]
    data_dir = PROJECT_ROOT / paths["data"]

    target_filename = paths.get("dataset_file", "small_recipes.csv")

    vector_store = _initialize_store(chroma_path)

    df = _load_recipe_data(data_dir, target_filename)

    documents = _transform_data_to_documents(df)

    if documents:
        _run_indexing_pipeline(documents, vector_store, config['embeddings_model'], api_token)
    else:
        logger.warning("No documents to index. Pipeline skipped.")


if __name__ == "__main__":
    ingest_csv_data()
