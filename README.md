## Simple RAG pipeline

# V1

PDF upload
|
v
PDF Parser
(Stores it in JSON format)
|
v
Chunker(Langchain Text-Splitter)
|
v
Embeddings(all-MiniLM-L6-v2)(Sentence Transformer)
|
v
Stored in ChromaDB
|
v
Retriever(ChromaDB K nearest neighbours)
|
v
Prompt builder
|
v
LLM genration
