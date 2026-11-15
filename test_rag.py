
from src.vector_store import load_vector_store
from src.retrieval import answer_question


# 1. Load the existing FAISS database
print("\n========== LOADING FAISS DATABASE ==========")

vector_store = load_vector_store()


# 2. Define test questions
test_questions = [
    {
        "category": "Relevant question - Goa",
        "question": "What is the best time to visit Goa?"
    },
    {
        "category": "Relevant question - Meghalaya",
        "question": "What are the popular attractions in Meghalaya?"
    },
    {
        "category": "Unsupported question",
        "question": "What is the population of Goa?"
    }
]


# 3. Execute each test
for test in test_questions:

    print("\n")
    print("=" * 60)
    print(f"TEST: {test['category']}")
    print("=" * 60)

    question = test["question"]

    print(f"\nQuestion: {question}")

    try:

        result = answer_question(
            vector_store=vector_store,
            question=question,
            k=4
        )

        # 4. Display generated answer
        print("\n========== GENERATED ANSWER ==========")
        print(result["answer"])

        # 5. Display source references
        print("\n========== SOURCE REFERENCES ==========")

        if result["sources"]:

            for source in result["sources"]:

                print(f"\nSource ID: {source['source_id']}")
                print(f"Filename: {source['source']}")
                print(f"Chunk: {source['chunk_index']}")

                print(
                    f"Content: {source['content'][:200]}"
                )

        else:
            print("No source documents retrieved.")
        print("\n========== CITATION VALIDATION ==========")

        validation = result["citation_validation"]

        if validation is not None:
            print("Is valid:", validation["is_valid"])
            print("Cited IDs:", validation["cited_ids"])
            print("Invalid IDs:", validation["invalid_ids"])
        else:
            print("Citation validation skipped (no sources retrieved).")

    except Exception as error:

        print("\nTEST FAILED!")
        print(type(error).__name__)
        print(str(error))


print("\n========== ALL TESTS COMPLETED ==========")
print("\n========== CHAT HISTORY TEST ==========")

chat_history = [
    {
        "role": "user",
        "content": "What is the best time to visit Goa?"
    },
    {
        "role": "assistant",
        "content": "November to February is generally the best time to visit Goa."
    }
]

question = "What about the budget?"

result = answer_question(
    vector_store=vector_store,
    question=question,
    k=4,
    chat_history=chat_history
)

print("\nQuestion:", question)
print("\nAnswer:", result["answer"])

print("\nCitation validation:")
print(result["citation_validation"])