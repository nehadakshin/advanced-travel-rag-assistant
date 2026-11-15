
from src.vector_store import get_or_create_vector_store

print("\n========== TEST: LOAD OR CREATE FAISS ==========")

vector_store = get_or_create_vector_store()

print("\nTest completed!")
print("Total indexed vectors:", vector_store.index.ntotal)
