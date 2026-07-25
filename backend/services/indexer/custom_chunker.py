from pathlib import Path
import json


##Chunked using chars instead of tokens
##For Practice and Logic

def load_documents(path: Path) -> list:
    with path.open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks


def chunk_documents(documents: list, chunk_size: int = 500, overlap: int = 100) -> list:
    chunks = []
    chunk_id = 0

    for document in documents:
        for page in document["pages"]:
            page_chunks = chunk_text(
                page["content"],
                chunk_size,
                overlap,
            )

            for chunk in page_chunks:
                chunks.append(
                    {
                        "document_name": document["document_name"],
                        "page_number": page["page_number"],
                        "chunk_id": chunk_id,
                        "text": chunk,
                    }
                )

                chunk_id += 1

    return chunks


def save_chunks(chunks: list, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as json_file:
        json.dump(chunks, json_file, indent=4)


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent

    INPUT_PATH = BASE_DIR / "parsed_documents.json"
    OUTPUT_PATH = BASE_DIR / "chunks.json"

    documents = load_documents(INPUT_PATH)

    chunks = chunk_documents(documents)

    save_chunks(chunks, OUTPUT_PATH)

    print(f"Created {len(chunks)} chunks.")