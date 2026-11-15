from src.document_tracker import (
    get_document_hashes,
    load_manifest,
    save_manifest
)

print("\n========== MANIFEST TEST ==========")

# Step 1: Calculate current document hashes
current_hashes = get_document_hashes("documents")

# Step 2: Save hashes
save_manifest(current_hashes)

# Step 3: Load manifest
manifest = load_manifest()

# Step 4: Display saved data
print("\n========== SAVED MANIFEST ==========")

for filename, file_hash in manifest["documents"].items():
    print(f"{filename}: {file_hash}")

print("\nTotal documents:", len(manifest["documents"]))