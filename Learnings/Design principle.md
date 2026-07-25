# SOFTWARE DESIGN PRINCIPLES

##1.Modularity
Break a large system into small independent pieces (modules).
Example:

Parser Module
│
├── parse_pdf()
└── parse_directory()

Chunking Module
│
└── chunk_document()

Embedding Module
│
└── generate_embeddings()

Retrieval Module
│
└── retrieve_chunks()

##2.Single Responsibility
Inside each module, every function should have one job
parse_pdf()->Parse one PDF.
parse_directory()->Find PDFs and call parse_pdf().

CHROMADB is a vectorBD
It contains of ##** collections ** and documents sort of like mongodb
But it amin task is to store vectors
It contains the follwing data:
id , text, vector, metadata
