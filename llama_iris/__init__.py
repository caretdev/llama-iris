from llama_iris.vectorstore import IRISVectorStore

from llama_index.vector_stores.loading import LOADABLE_VECTOR_STORES

LOADABLE_VECTOR_STORES[IRISVectorStore.class_name()] = IRISVectorStore

__all__ = ["IRISVectorStore"]
