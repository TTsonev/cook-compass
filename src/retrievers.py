from abc import ABC, abstractmethod
from typing import List, Any

from haystack import Pipeline, Document
from haystack.components.embedders import HuggingFaceAPITextEmbedder
from haystack.utils import Secret
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever


class BaseRetrieverPipeline(ABC):
    @abstractmethod
    def retrieve_documents(self, query: str) -> List[Document]:
        pass


class VectorRetrieverPipeline(BaseRetrieverPipeline):
    def __init__(self, document_store: Any, embeddings_model: str, api_token: str, top_k: int = 10):
        """        
        Args:
            document_store: The document store to retrieve from.
            embeddings_model: The name of the embeddings model to use.
            api_token: The API token for the embeddings model.
            top_k: The number of documents to retrieve.
        """
        self.document_store = document_store
        self.embeddings_model = embeddings_model
        self.api_token = api_token
        self.top_k = top_k
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self) -> Pipeline:
        text_embedder = HuggingFaceAPITextEmbedder(
            api_type="serverless_inference_api",
            api_params={"model": self.embeddings_model},
            token=Secret.from_token(self.api_token)
        )

        pipeline = Pipeline()
        pipeline.add_component("query_embedder", text_embedder)
        pipeline.add_component("retriever", ChromaEmbeddingRetriever(self.document_store, top_k=self.top_k))
        pipeline.connect("query_embedder.embedding", "retriever.query_embedding")
        return pipeline

    def retrieve_documents(self, query: str) -> List[Document]:
        results = self.pipeline.run({"query_embedder": {"text": query}})
        return results["retriever"]["documents"]


class HybridRetriever(BaseRetrieverPipeline):
    # TODO: implement (https://haystack.deepset.ai/tutorials/33_hybrid_retrieval)
    pass