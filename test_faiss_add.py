from src.vector_store import load_vector_store
from langchain_core.documents import Document

TEST_PATH = "vector_db/faiss_delete_test"

print("\n========== FAISS PERSISTENCE TEST ==========")

# 1. Load test database
vector_store = load_vector_store(TEST_PATH)
print("Initial vectors:", vector_store.index.ntotal)

# 2. Add a temporary document
new_document = Document(
    page_content="Persistence testing for the travel RAG project.",
    metadata={
        "source": "documents/persistence_test.txt",
        "chunk_index": 0
    }
)

vector_store.add_documents([new_document])

print("Vectors after addition:", vector_store.index.ntotal)

# 3. Save updated database
vector_store.save_local(TEST_PATH)
print("Updated database saved.")

# 4. Load database again
reloaded_store = load_vector_store(TEST_PATH)

print("Vectors after reloading:", reloaded_store.index.ntotal)

# 5. Verify document exists
results = reloaded_store.similarity_search(
    "persistence testing",
    k=1
)

print("\nRetrieved content:")
print(results[0].page_content)
print("Source:", results[0].metadata.get("source"))

print("\n========== TEST COMPLETED ==========")