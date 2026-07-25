from backend.services.querying.retriever import (
    load_embedding_model,
    load_collection,
    retrieve_documents,
)

from backend.services.querying.prompt_builder import build_prompt

from backend.services.querying.LLM import (
    load_client,
    generate_response,
)


def run() -> None:
    """Run the complete RAG pipeline."""

    embedding_model = load_embedding_model()
    collection = load_collection()
    llm_client = load_client()

    print("=" * 50)
    print("AI Research Assistant")
    print("Type 'exit' to quit.")
    print("=" * 50)

    while True:
        question = input("\nQuestion: ").strip()

        if question.lower() == "exit":
            print("\nGoodbye!")
            break

        retrieved_chunks = retrieve_documents(
            question=question,
            model=embedding_model,
            collection=collection,
        )

        prompt = build_prompt(
            question=question,
            retrieved_chunks=retrieved_chunks,
        )

        answer = generate_response(
            prompt=prompt,
            client=llm_client,
        )

        print("\nAnswer:")
        print("-" * 50)
        print(answer)
        print("-" * 50)


if __name__ == "__main__":
    run()