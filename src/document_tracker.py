
import hashlib
from pathlib import Path


def calculate_file_hash(file_path):
    """
    Calculate SHA-256 hash of a file.

    Returns:
        Hexadecimal hash string.
    """

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        # Read the file in chunks to avoid loading
        # the entire file into memory.
        for file_chunk in iter(
            lambda: file.read(4096),
            b""
        ):
            sha256.update(file_chunk)

    return sha256.hexdigest()


def get_document_hashes(directory="documents"):
    """
    Calculate hashes for all supported documents.

    Returns:
        Dictionary containing filename and hash.
    """

    documents_path = Path(directory)

    if not documents_path.exists():
        raise FileNotFoundError(
            f"Directory not found: {directory}"
        )

    document_hashes = {}

    for file_path in sorted(documents_path.iterdir()):

        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in [".pdf", ".txt"]:
            continue

        file_hash = calculate_file_hash(file_path)

        document_hashes[file_path.name] = file_hash

        print(f"{file_path.name}: {file_hash}")

    return document_hashes
import json

MANIFEST_PATH = Path("metadata/document_manifest.json")


def load_manifest():
    """Load the existing document manifest."""

    if not MANIFEST_PATH.exists():
        return {"documents": {}}

    with open(MANIFEST_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def save_manifest(document_hashes):
    """Save the latest document hashes to the manifest."""

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "documents": document_hashes
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=4)

    print("Document manifest saved successfully.")
def detect_document_changes(directory="documents"):
    """
    Compare current document hashes with the saved manifest.
    Identify new, modified, unchanged and deleted documents.
    """

    # Get previously saved hashes
    manifest = load_manifest()
    old_hashes = manifest["documents"]

    # Calculate hashes of current files
    current_hashes = get_document_hashes(directory)

    changes = {
        "new": [],
        "modified": [],
        "unchanged": [],
        "deleted": []
    }

    # Identify new, modified and unchanged files
    for filename, current_hash in current_hashes.items():

        if filename not in old_hashes:
            changes["new"].append(filename)

        elif old_hashes[filename] != current_hash:
            changes["modified"].append(filename)

        else:
            changes["unchanged"].append(filename)

    # Identify deleted files
    for filename in old_hashes:

        if filename not in current_hashes:
            changes["deleted"].append(filename)

    return changes