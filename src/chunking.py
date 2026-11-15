
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents,
    chunk_size=500,
    chunk_overlap=50
):
    """
    Split LangChain Document objects into smaller chunks.

    Args:
        documents: List of LangChain Document objects.
        chunk_size: Maximum chunk length in characters.
        chunk_overlap: Number of overlapping characters.

    Returns:
        List of smaller LangChain Document objects.
    """

    # Create the text splitter.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
        add_start_index=True
    )

    chunks = []

    # Split each document separately.
    for document in documents:

        document_chunks = text_splitter.split_documents(
            [document]
        )

        # Add a chunk number to each chunk's metadata.
        for index, chunk in enumerate(document_chunks):
            chunk.metadata["chunk_index"] = index

        chunks.extend(document_chunks)

    print(f"Original documents: {len(documents)}")
    print(f"Generated chunks: {len(chunks)}")

    return chunks
