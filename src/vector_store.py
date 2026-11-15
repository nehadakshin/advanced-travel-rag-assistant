
from pathlib import Path

from langchain_community.vectorstores import FAISS

from src.embeddings import GeminiRetrievalEmbeddings


DEFAULT_INDEX_PATH = "vector_db/travel_faiss"


def create_vector_store(
    chunks,
    save_path=DEFAULT_INDEX_PATH
):
    """
    Create a FAISS vector store from document chunks.
    Generate embeddings, index the vectors and save locally.
    """

    if not chunks:
        raise ValueError("No document chunks provided.")

    # Initialize the Gemini embedding interface.
    embeddings_model = GeminiRetrievalEmbeddings()

    print("Creating FAISS vector store...")

    # Convert chunks into embeddings and create the FAISS index.
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        normalize_L2=True
    )

    print("FAISS index created successfully!")
    print(f"Total indexed vectors: {vector_store.index.ntotal}")

    # Create the parent directory if needed.
    Path(save_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save the FAISS index and associated documents.
    vector_store.save_local(save_path)

    print(f"FAISS index saved at: {save_path}")

    return vector_store


def load_vector_store(
    save_path=DEFAULT_INDEX_PATH
):
    """
    Load an existing FAISS index from local storage.
    """

    if not Path(save_path).exists():
        raise FileNotFoundError(
            f"FAISS index not found at: {save_path}"
        )

    embeddings_model = GeminiRetrievalEmbeddings()

    vector_store = FAISS.load_local(
        folder_path=save_path,
        embeddings=embeddings_model,
        allow_dangerous_deserialization=True,
        normalize_L2=True
    )

    print("FAISS index loaded successfully!")
    print(f"Total indexed vectors: {vector_store.index.ntotal}")

    return vector_store


def search_similar_chunks(
    vector_store,
    question,
    k=3
):
    """
    Retrieve the top-k chunks relevant to a question.
    """

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    results = vector_store.similarity_search(
        query=question,
        k=k
    )

    print(f"\nQuestion: {question}")
    print(f"Retrieved chunks: {len(results)}")

    for index, document in enumerate(results):

        print(f"\n--- RESULT {index + 1} ---")
        print(f"Source: {document.metadata.get('source')}")
        print(f"Chunk: {document.metadata.get('chunk_index')}")
        print(f"Content:\n{document.page_content}")

    return results

def get_or_create_vector_store(
    chunks=None,
    save_path=DEFAULT_INDEX_PATH
):
    """
    Load an existing FAISS index if available.
    Otherwise, create a new index using document chunks.
    """

    index_path = Path(save_path)

    # Check whether the saved FAISS index exists.
    if (
        (index_path / "index.faiss").exists()
        and (index_path / "index.pkl").exists()
    ):

        print("Existing FAISS database found.")
        print("Loading saved vector store...")

        return load_vector_store(save_path)

    # No saved index exists.
    print("No existing FAISS database found.")

    if not chunks:
        raise ValueError(
            "Document chunks are required to create a new FAISS index."
        )

    print("Creating a new FAISS database...")

    return create_vector_store(
        chunks=chunks,
        save_path=save_path
    )
def delete_documents_from_vector_store(
    vector_store,
    source_files
):
    """
    Delete all FAISS vectors belonging to specific source files.
    """

    if not source_files:
        print("No documents to delete.")
        return 0

    source_files = {
        str(Path(source).as_posix())
        for source in source_files
    }

    ids_to_delete = []

    # Find vectors belonging to the specified documents.
    for doc_id in list(vector_store.index_to_docstore_id.values()):

        document = vector_store.docstore.search(doc_id)

        if document is None:
            continue

        source = str(
            Path(document.metadata.get("source", "")).as_posix()
        )

        if source in source_files:
            ids_to_delete.append(doc_id)

    if not ids_to_delete:
        print("No matching vectors found.")
        return 0

    print(f"Vectors to delete: {len(ids_to_delete)}")

    # Delete the identified vectors.
    vector_store.delete(ids=ids_to_delete)

    print("Document vectors deleted successfully.")
    print("Remaining vectors:", vector_store.index.ntotal)

    return len(ids_to_delete)