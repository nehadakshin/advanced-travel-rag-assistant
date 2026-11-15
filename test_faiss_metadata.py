from src.vector_store import load_vector_store

print("\n========== FAISS METADATA INSPECTION ==========")

vector_store = load_vector_store()

print("\nTotal vectors:", vector_store.index.ntotal)

print("\nIndex-to-document ID mapping:")

for index, doc_id in vector_store.index_to_docstore_id.items():

    document = vector_store.docstore.search(doc_id)

    print(f"\nVector index: {index}")
    print(f"Document ID: {doc_id}")
    print(f"Source: {document.metadata.get('source')}")
    print(f"Chunk: {document.metadata.get('chunk_index')}")

print("\n========== INSPECTION COMPLETED ==========")