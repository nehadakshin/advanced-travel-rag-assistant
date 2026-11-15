
import json
import os
import shutil
import tempfile
from pathlib import Path

from src.document_tracker import (
    detect_document_changes,
    get_document_hashes,
    MANIFEST_PATH,
)
from src.ingestion import load_documents, load_selected_documents
from src.chunking import split_documents
from src.vector_store import (
    create_vector_store,
    load_vector_store,
    delete_documents_from_vector_store,
    DEFAULT_INDEX_PATH,
)


def _create_staged_manifest(document_hashes):
    """
    Write the new manifest to a temporary file in the same directory.
    It will replace the production manifest only during commit.
    """
    manifest_path = Path(MANIFEST_PATH)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    file_descriptor, temp_name = tempfile.mkstemp(
        prefix="document_manifest_",
        suffix=".tmp",
        dir=str(manifest_path.parent),
    )

    temp_path = Path(temp_name)

    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
            json.dump(
                {"documents": document_hashes},
                file,
                indent=4,
            )
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    return temp_path


def _commit_staged_changes(
    staging_path,
    index_path,
    staged_manifest_path,
):
    """
    Replace the production index and manifest.

    If any part of the commit fails, restore the previous versions.
    """
    index_path = Path(index_path)
    manifest_path = Path(MANIFEST_PATH)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    backup_index = Path(
        tempfile.mkdtemp(
            prefix=f".{index_path.name}_backup_",
            dir=str(index_path.parent),
        )
    )
    backup_index.rmdir()

    backup_manifest = manifest_path.with_name(
        f".{manifest_path.name}.backup"
    )

    index_was_backed_up = False
    manifest_was_backed_up = False
    new_index_installed = False
    new_manifest_installed = False

    try:
        # Back up the existing index.
        if index_path.exists():
            os.replace(index_path, backup_index)
            index_was_backed_up = True

        # Install the staged index.
        os.replace(staging_path, index_path)
        new_index_installed = True

        # Back up the existing manifest.
        if manifest_path.exists():
            if backup_manifest.exists():
                backup_manifest.unlink()

            os.replace(manifest_path, backup_manifest)
            manifest_was_backed_up = True

        # Install the new manifest.
        os.replace(staged_manifest_path, manifest_path)
        new_manifest_installed = True

    except Exception:
        # Roll back the manifest first.
        if new_manifest_installed and manifest_path.exists():
            manifest_path.unlink()

        if manifest_was_backed_up and backup_manifest.exists():
            os.replace(backup_manifest, manifest_path)

        # Roll back the index.
        if new_index_installed and index_path.exists():
            shutil.rmtree(index_path)

        if index_was_backed_up and backup_index.exists():
            os.replace(backup_index, index_path)

        raise

    else:
        # Commit succeeded. Old versions are no longer needed.
        if backup_index.exists():
            shutil.rmtree(backup_index)

        if backup_manifest.exists():
            backup_manifest.unlink()

    finally:
        # Remove any uncommitted temporary files.
        if Path(staging_path).exists():
            shutil.rmtree(staging_path)

        if Path(staged_manifest_path).exists():
            Path(staged_manifest_path).unlink()


def run_incremental_indexing(
    documents_directory="documents",
    index_path=DEFAULT_INDEX_PATH,
):
    print("\n========== INCREMENTAL INDEXING ==========")

    documents_directory = Path(documents_directory)
    index_path = Path(index_path)
    manifest_path = Path(MANIFEST_PATH)

    changes = detect_document_changes(str(documents_directory))

    new_files = changes["new"]
    modified_files = changes["modified"]
    deleted_files = changes["deleted"]
    unchanged_files = changes["unchanged"]

    index_exists = (
        (index_path / "index.faiss").exists()
        and (index_path / "index.pkl").exists()
    )

    manifest_exists = manifest_path.exists()

    # If either the index or manifest is missing, rebuild the index
    # from all available documents in a separate staging directory.
    initial_indexing = not index_exists or not manifest_exists

    index_path.parent.mkdir(parents=True, exist_ok=True)

    staging_path = Path(
        tempfile.mkdtemp(
            prefix=f".{index_path.name}_staging_",
            dir=str(index_path.parent),
        )
    )

    staged_manifest_path = None

    try:
        if initial_indexing:
            print("\nInitial indexing or recovery rebuild...")

            documents = load_documents(str(documents_directory))

            if not documents:
                raise ValueError(
                    "No supported PDF or TXT documents found."
                )

            chunks = split_documents(documents)

            if not chunks:
                raise ValueError(
                    "No document chunks were generated."
                )

            vector_store = create_vector_store(
                chunks,
                save_path=str(staging_path),
            )

            current_hashes = get_document_hashes(
                str(documents_directory)
            )

            stats = {
                "new_documents": len(current_hashes),
                "modified_documents": 0,
                "deleted_documents": 0,
                "unchanged_documents": 0,
                "added_chunks": len(chunks),
                "deleted_vectors": 0,
                "total_vectors": vector_store.index.ntotal,
            }

        else:
            print("\nChecking document changes...")

            print(f"New documents: {len(new_files)}")
            print(f"Modified documents: {len(modified_files)}")
            print(f"Deleted documents: {len(deleted_files)}")
            print(f"Unchanged documents: {len(unchanged_files)}")

            # No changes: load the production index without modifying it.
            if not new_files and not modified_files and not deleted_files:
                shutil.rmtree(staging_path)

                vector_store = load_vector_store(str(index_path))

                stats = {
                    "new_documents": 0,
                    "modified_documents": 0,
                    "deleted_documents": 0,
                    "unchanged_documents": len(unchanged_files),
                    "added_chunks": 0,
                    "deleted_vectors": 0,
                    "total_vectors": vector_store.index.ntotal,
                }

                print("\nKnowledge base is already up to date.")
                return {
                    "vector_store": vector_store,
                    "stats": stats,
                }

            # Copy the production index. All modifications happen
            # on this copy, not on the production index.
            shutil.rmtree(staging_path)
            shutil.copytree(index_path, staging_path)

            vector_store = load_vector_store(str(staging_path))

            # Modified documents must have their old vectors removed.
            # Deleted documents must also have their old vectors removed.
            files_to_delete = [
                str(documents_directory / filename)
                for filename in modified_files + deleted_files
            ]

            deleted_vectors = delete_documents_from_vector_store(
                vector_store,
                files_to_delete,
            )

            # Load only new and modified documents.
            files_to_load = [
                str(documents_directory / filename)
                for filename in new_files + modified_files
            ]

            documents = (
                load_selected_documents(files_to_load)
                if files_to_load
                else []
            )

            chunks = (
                split_documents(
                    documents,
                    chunk_size=500,
                    chunk_overlap=50,
                )
                if documents
                else []
            )

            if chunks:
                vector_store.add_documents(chunks)

            # Persist the staged index.
            vector_store.save_local(str(staging_path))

            current_hashes = get_document_hashes(
                str(documents_directory)
            )

            stats = {
                "new_documents": len(new_files),
                "modified_documents": len(modified_files),
                "deleted_documents": len(deleted_files),
                "unchanged_documents": len(unchanged_files),
                "added_chunks": len(chunks),
                "deleted_vectors": deleted_vectors,
                "total_vectors": vector_store.index.ntotal,
            }

        # Prepare the manifest before committing either file.
        staged_manifest_path = _create_staged_manifest(
            current_hashes
        )

        # Install the staged index and manifest together,
        # with rollback if the commit fails.
        _commit_staged_changes(
            staging_path=staging_path,
            index_path=index_path,
            staged_manifest_path=staged_manifest_path,
        )

        print("\n========== INDEXING SUMMARY ==========")
        for key, value in stats.items():
            print(f"{key}: {value}")

        print("\nIncremental indexing completed successfully!")

        return {
            "vector_store": vector_store,
            "stats": stats,
        }

    except Exception as error:
        print("\nIncremental indexing failed.")
        print(f"Reason: {error}")
        print("The previous production index has been preserved.")
        raise

    finally:
        # Clean up temporary staging directory if it still exists.
        if staging_path.exists():
            shutil.rmtree(staging_path)

        if (
            staged_manifest_path is not None
            and Path(staged_manifest_path).exists()
        ):
            Path(staged_manifest_path).unlink()
