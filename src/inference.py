import logging
import os
import queue
import threading
from typing import List, Generator, Tuple, Any

from haystack.components.generators.chat import HuggingFaceAPIChatGenerator
from haystack.dataclasses import ChatMessage, StreamingChunk, Document
from haystack.utils import Secret
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

from .ingest import ingest_csv_data
from .prompts import intro_prompt, rewrrite_query_prompt, keywords_prompt
from .retrievers import VectorRetrieverPipeline
from .utils import PROJECT_ROOT, load_config
from .keywords import KEYWORDS

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger("Inference")


def _load_and_validate_config() -> Tuple[Any, str]:
    """Loads config and ensures API token is present."""
    api_token = os.getenv("API_TOKEN")
    if not api_token:
        logger.critical("API_TOKEN not found in environment. Cannot proceed.")
        raise ValueError("Missing API_TOKEN")

    config = load_config()
    return config, api_token


class InferenceEngine:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    @classmethod
    def from_config(cls):
        config, api_token = _load_and_validate_config()
        chroma_path = PROJECT_ROOT / config["paths"]["chroma"]

        vector_store = ChromaDocumentStore(persist_path=str(chroma_path))

        if not vector_store.count_documents():
            logger.info("Document store is empty. Triggering ingestion...")
            ingest_csv_data()
        else:
            logger.info("Document store found with %d documents.", vector_store.count_documents())

        retriever = VectorRetrieverPipeline(
            document_store=vector_store,
            embeddings_model=config['embeddings_model'],
            api_token=api_token,
            top_k=20,
        )

        llm = HuggingFaceAPIChatGenerator(
            api_type="serverless_inference_api",
            api_params={"model": config['llm']},
            token=Secret.from_token(api_token) if api_token else None
        )

        return cls(retriever, llm)

    def _generate_search_query(self, message_history: List[dict]) -> str:
        last_user_msg = message_history[-1]['content']

        if len(message_history) < 2:
            return last_user_msg

        history_subset = message_history[-5:]
        conversation_text = ""
        for msg in history_subset:
            role = "User" if msg['role'] == 'user' else "Assistant"
            conversation_text += f"{role}: {msg['content']}\n"

        rewrite_prompt = rewrrite_query_prompt.format(
            last_user_msg=last_user_msg,
            conversation_text=conversation_text
        )

        original_callback = self.llm.streaming_callback
        self.llm.streaming_callback = None

        try:
            response = self.llm.run([ChatMessage.from_user(rewrite_prompt)])
            rewritten_query = response["replies"][0].text.strip()
            print(f"REWR: {rewritten_query}")

            logger.info(f"Original Query: '{last_user_msg}' -> Rewritten: '{rewritten_query}'")
            return rewritten_query
        except Exception as e:
            logger.error(f"Error when rewriting: {e}")
            return last_user_msg
        finally:
            self.llm.streaming_callback = original_callback
    
    def _extract_keywords(self, query: str) -> List[str]:
        prompt = keywords_prompt.format(
            query=query,
            available_keywords=', '.join(KEYWORDS)
        )

        original_callback = self.llm.streaming_callback
        self.llm.streaming_callback = None

        try:
            response = self.llm.run([ChatMessage.from_user(prompt)])
            content = response["replies"][0].text.strip()
            
            if not content:
                return []
                
            found_keywords = [k.strip() for k in content.split(',')]
            # Validate
            valid_keywords = [k for k in found_keywords if k in KEYWORDS]
            return valid_keywords
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
        finally:
            self.llm.streaming_callback = original_callback


    def _build_messages(self, query: str, context_docs: List[Document], message_history: List[dict]) -> List[
        ChatMessage]:
        context_parts = []
        for doc in context_docs:
            part = (
                f"--- Recipe Option ---\n"
                f"Name: {doc.meta.get('name')}\n"
                f"{doc.content}\n"
                f"Instructions: {doc.meta.get('steps')}\n"
            )
            context_parts.append(part)

        full_context_string = "\n".join(context_parts)

        formatted_system_prompt = intro_prompt.format(context_content=full_context_string, user_message=query)

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

        last_user_message = message_history[-1]['content']

        search_query = self._generate_search_query(message_history)
        
        logger.info(f"Search Query: {search_query}")
        context_docs = self.retriever.retrieve_documents(search_query)

        extracted_keywords = self._extract_keywords(search_query)
        logger.info(f"Extracted keywords: {extracted_keywords}")

        if extracted_keywords:
            filter_docs = [
                doc for doc in context_docs 
                if all(k in doc.meta.get('tags', []) for k in extracted_keywords)
            ]
            
            if filter_docs:
                logger.info(f"Filtered down to {len(filter_docs)} documents based on keywords.")
                context_docs = filter_docs

        logger.info(f'Context: {[doc.meta.get('name', 'unknown') for doc in context_docs]}')
        prompt = self._build_messages(last_user_message, context_docs, message_history[:-1])

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
