# Llama-index with InterSystems IRIS

[Llama-index](https://github.com/run-llama/llama_index) with support for InterSystems IRIS

## Install

```shell
pip install llama-iris
```

## Example

```python
import os
from dotenv import load_dotenv

from llama_index import SimpleDirectoryReader, StorageContext, ServiceContext
from llama_index.indices.vector_store import VectorStoreIndex
import openai

from llama_iris import IRISVectorStore


load_dotenv(override=True)

documents = SimpleDirectoryReader("./data/paul_graham").load_data()
print("Document ID:", documents[0].doc_id)

vector_store = IRISVectorStore.from_params(
    connection_string=CONNECTION_STRING,
    table_name="paul_graham_essay",
    embed_dim=1536,  # openai embedding dimension
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents, 
    storage_context=storage_context, 
    show_progress=True, 
)
query_engine = index.as_query_engine()

response = query_engine.query("What did the author do?")

import textwrap
print(textwrap.fill(str(response), 100))
```