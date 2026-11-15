from src.document_tracker import detect_document_changes
from src.ingestion import load_selected_documents

print("\n========== INCREMENTAL DETECTION TEST ==========")

# Step 1: Detect changes
changes = detect_document_changes()

print("\nNew files:", changes["new"])
print("Modified files:", changes["modified"])
print("Unchanged files:", changes["unchanged"])
print("Deleted files:", changes["deleted"])

# Step 2: Select only new and modified files
from pathlib import Path

files_to_load = [
    str(Path("documents") / filename)
    for filename in changes["new"] + changes["modified"]
]
print("\nFiles selected for loading:", files_to_load)

# Step 3: Load only selected files
if files_to_load:
    documents = load_selected_documents(files_to_load)
    print("\nLoaded document objects:", len(documents))
else:
    print("\nNo new or modified documents to load.")

print("\n========== TEST COMPLETED ==========")