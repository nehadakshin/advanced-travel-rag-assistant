
from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)


def load_documents(directory="documents"):
    """
    Load all supported PDF and TXT files from a directory.

    Returns:
        A list of LangChain Document objects.
    """

    documents = []
    documents_path = Path(directory)

    # Check whether the documents directory exists.
    if not documents_path.exists():
        raise FileNotFoundError(
            f"Directory not found: {documents_path}"
        )

    # Process every file in the directory.
    for file_path in sorted(documents_path.iterdir()):

        if not file_path.is_file():
            continue

        print(f"Loading: {file_path.name}")

        # Load PDF files.
        if file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))
            loaded_documents = loader.load()

        # Load TXT files.
        elif file_path.suffix.lower() == ".txt":
            loader = TextLoader(
                str(file_path),
                encoding="utf-8"
            )
            loaded_documents = loader.load()

        else:
            print(f"Skipping unsupported file: {file_path.name}")
            continue

        documents.extend(loaded_documents)

        print(
            f"  Extracted {len(loaded_documents)} "
            f"document(s) from {file_path.name}"
        )

    print(f"\nTotal documents loaded: {len(documents)}")

    return documents
def load_selected_documents(file_paths):
    """
    Load only the specified PDF and TXT files.

    Args:
        file_paths: List of file paths to load.

    Returns:
        A list of LangChain Document objects.
    """

    documents = []

    for file_path in file_paths:

        file_path = Path(file_path)

        if not file_path.exists():
            print(f"File not found: {file_path}")
            continue

        print(f"Loading selected file: {file_path.name}")

        if file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))
            loaded_documents = loader.load()

        elif file_path.suffix.lower() == ".txt":
            loader = TextLoader(
                str(file_path),
                encoding="utf-8"
            )
            loaded_documents = loader.load()

        else:
            print(f"Skipping unsupported file: {file_path.name}")
            continue

        documents.extend(loaded_documents)

        print(
            f"  Extracted {len(loaded_documents)} "
            f"document(s) from {file_path.name}"
        )

    print(f"\nTotal selected documents loaded: {len(documents)}")

    return documents