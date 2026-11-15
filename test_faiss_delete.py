from pathlib import Path
import shutil

from src.vector_store import (
    load_vector_store,
    delete_documents_from_vector_store
)

print("\n========== FAISS SAFE DELETE TEST ==========")

# Step 1: Load original database
original_store = load_vector_store()

print("Original vector count:", original_store.index.ntotal)

# Step 2: Create a separate physical copy of the FAISS files
test_path = "vector_db/faiss_delete_test"

shutil.rmtree(test_path, ignore_errors=True)

shutil.copytree(
    "vector_db/travel_faiss",
    test_path
)

# Step 3: Load the independent test copy
test_store = load_vector_store(test_path)

print("Test copy created successfully.")
print("Test copy vectors:", test_store.index.ntotal)

# Step 4: Delete Goa guide vectors from test copy
deleted_count = delete_documents_from_vector_store(
    test_store,
    ["documents/goa_guide.txt"]
)

# Step 5: Verify
remaining_goa_vectors = 0

for doc_id in test_store.index_to_docstore_id.values():

    document = test_store.docstore.search(doc_id)

    if document.metadata.get("source") == "documents/goa_guide.txt":
        remaining_goa_vectors += 1

print("\n========== VERIFICATION ==========")

print("Deleted vectors:", deleted_count)
print("Original vectors:", original_store.index.ntotal)
print("Test copy vectors:", test_store.index.ntotal)
print("Remaining Goa guide vectors:", remaining_goa_vectors)

print("\n========== TEST COMPLETED ==========")