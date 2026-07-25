
def build_prompt(question:str, retrieved_chunks:list)->str:
    context=""

    for chunk in retrieved_chunks:
        context+= (
            f"Document: {chunk['document_name']}\n"
            f"Page:{chunk['page_number']}\n\n"
            f"{chunk['text']}\n\n"
        )

    prompt= f"""
    You are a helpful AI assistant.

    Answer the user's question using ONLY the context below.
    If the answer is not present in the context, reply:
    "I dont have enough information to answer that."

    Context
    --------
    {context}
    --------
    Question:

    {question}

    Answer:

"""
    return prompt
