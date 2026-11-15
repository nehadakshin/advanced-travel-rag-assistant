from src.document_tracker import get_document_hashes

print("\n========== DOCUMENT HASH TEST ==========")

hashes = get_document_hashes("documents")

print("\n========== RESULTS ==========")

for filename, file_hash in hashes.items():

    print(f"\nFilename: {filename}")
    print(f"Hash: {file_hash}")

print("\nTotal documents:", len(hashes))