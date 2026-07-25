from pathlib import Path
import json

from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(path: Path) -> list:
    with path.open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


def chunk_documents(documents: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )

    chunks = []
    chunk_id = 0

    for document in documents:
        for page in document["pages"]:

            page_chunks = splitter.split_text(page["content"])

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