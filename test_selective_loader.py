from src.ingestion import load_selected_documents

print("\n========== SELECTIVE LOADER TEST ==========")

selected_files = [
    "documents/goa_guide.txt",
    "documents/incremental_test.txt"
]

documents = load_selected_documents(selected_files)

print("\n========== VERIFICATION ==========")

print("Total loaded document objects:", len(documents))

for document in documents:
    print("\nSource:", document.metadata.get("source"))
    print("Content:", document.page_content[:150])

print("\n========== TEST COMPLETED ==========")