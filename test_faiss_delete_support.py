from src.vector_store import load_vector_store

print("\n========== FAISS DELETE SUPPORT TEST ==========")

vector_store = load_vector_store()

print("FAISS index type:", type(vector_store.index).__name__)

print("Initial vector count:", vector_store.index.ntotal)

print(
    "Delete method available:",
    hasattr(vector_store, "delete")
)

print(
    "Index-to-document mapping available:",
    hasattr(vector_store, "index_to_docstore_id")
)

print("\n========== TEST COMPLETED ==========")