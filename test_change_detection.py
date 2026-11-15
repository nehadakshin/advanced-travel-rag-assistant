from src.document_tracker import detect_document_changes

print("\n========== DOCUMENT CHANGE DETECTION ==========")

changes = detect_document_changes("documents")

for change_type, files in changes.items():

    print(f"\n{change_type.upper()} DOCUMENTS:")

    if files:
        for filename in files:
            print(f"  - {filename}")
    else:
        print("  None")

print("\n========== TEST COMPLETED ==========")