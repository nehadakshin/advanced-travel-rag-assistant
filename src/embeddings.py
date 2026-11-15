
import os

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings


# Load environment variables from .env
load_dotenv()


def get_embedding_model(task_type="RETRIEVAL_DOCUMENT"):
    """
    Initialize the Gemini embedding model.

    RETRIEVAL_DOCUMENT is used for embedding stored documents.
    RETRIEVAL_QUERY will be used for embedding user questions.
    """

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found. Check your .env file."
        )

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        task_type=task_type,
        google_api_key=api_key
    )

    return embeddings


def generate_embeddings(chunks):
    """
    Generate one embedding vector for each text chunk.

    Args:
        chunks: List of LangChain Document objects.

    Returns:
        List of numerical embedding vectors.
    """

    if not chunks:
        raise ValueError("No chunks provided for embedding.")

    # Initialize Gemini embedding model for documents.
    embeddings_model = get_embedding_model(
        task_type="RETRIEVAL_DOCUMENT"
    )

    # Extract text from each LangChain Document.
    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    print(f"Generating embeddings for {len(texts)} chunks...")

    # Convert all chunk texts into vectors.
    vectors = embeddings_model.embed_documents(texts)

    print("Embedding generation completed!")
    print(f"Total vectors: {len(vectors)}")
    print(f"Vector dimensions: {len(vectors[0])}")

    return vectors

def generate_query_embedding(question):
    """
    Convert a user question into an embedding vector.
    """

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    # Initialize Gemini for query retrieval.
    embeddings_model = get_embedding_model(
        task_type="RETRIEVAL_QUERY"
    )

    query_vector = embeddings_model.embed_query(question)

    print("Query embedding generated!")
    print(f"Vector dimensions: {len(query_vector)}")

    return query_vector

from langchain_core.embeddings import Embeddings


class GeminiRetrievalEmbeddings(Embeddings):
    """
    Use the correct Gemini task type for documents and queries.
    """

    def __init__(self):
        self.document_model = get_embedding_model(
            task_type="RETRIEVAL_DOCUMENT"
        )

        self.query_model = get_embedding_model(
            task_type="RETRIEVAL_QUERY"
        )

    def embed_documents(self, texts):
        return self.document_model.embed_documents(texts)

    def embed_query(self, text):
        return self.query_model.embed_query(text)
