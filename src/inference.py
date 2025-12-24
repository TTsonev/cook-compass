import os
import queue
import threading
import logging
from typing import List, Generator

from haystack.components.generators.chat import HuggingFaceAPIChatGenerator
from haystack.dataclasses import ChatMessage, StreamingChunk, Document
from haystack.utils import Secret
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

from .ingest import ingest_csv_data
from .retrievers import VectorRetrieverPipeline
from .prompts import intro_prompt
from .utils import PROJECT_ROOT, load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# TODO: logging is AI generated, might need some work

class InferenceEngine:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    @classmethod
    def from_config(cls):
        config = load_config()

        chroma_path = PROJECT_ROOT / config["paths"]["chroma"]
        
        api_token = os.getenv("API_TOKEN")

        vector_store = ChromaDocumentStore(persist_path=str(chroma_path))
        
        if not vector_store.count_documents():
            logger.info("Document store is empty.")
            ingest_csv_data()
        else:
            logger.info("Document store found with %d documents.", vector_store.count_documents())

        retriever = VectorRetrieverPipeline(
            document_store=vector_store,
            embeddings_model=config['embeddings_model'],
            api_token=api_token
        )

        llm = HuggingFaceAPIChatGenerator(
            api_type="serverless_inference_api",
            api_params={"model": config['llm']},
            token=Secret.from_token(api_token) if api_token else None
        )
        
        return cls(retriever, llm)

    def _build_messages(self, query: str, context_docs: List[Document], message_history: List[dict]) -> List[ChatMessage]:
        context_content = "\n".join([doc.content for doc in context_docs])
        formatted_system_prompt = intro_prompt.format(context_content=context_content)

        # TODO: create a nicer prompt template

        messages = [ChatMessage.from_system(formatted_system_prompt)]
        
        messages += [
            ChatMessage.from_user(message['content']) if message['role'] == 'user' else
            ChatMessage.from_assistant(message['content'])
            for message in message_history
        ]
        return messages

    def stream_response(self, message_history: List[dict]) -> Generator[str, None, None]:
        if not message_history:
            return
        
        query = message_history[-1]['content'] # last message in chat history is the current query
        
        # TODO: play around with different retrieval strategies
        context_docs = self.retriever.retrieve_documents(query)
        
        prompt = self._build_messages(query, context_docs, message_history[:-1])
        
        # output streaming (should be doable only with a callback but I couldn't find anything in the docs)
        q = queue.Queue()

        def callback(chunk: StreamingChunk):
            q.put(chunk.content)

        self.llm.streaming_callback = callback

        def run_llm():
            try:
                self.llm.run(prompt)
            finally:
                q.put(None)

        thread = threading.Thread(target=run_llm)
        thread.start()

        while True:
            token = q.get()
            if token is None:
                break
            yield token
        
        thread.join()

