from src.retrieval import validate_citations


# Test 1: All citations are valid
answer_1 = (
    "The best time to visit Goa is November to February [S1]. "
    "Goa has several popular beaches [S2]."
)

sources_1 = [
    {"source_id": "S1"},
    {"source_id": "S2"}
]


# Test 2: One invalid citation
answer_2 = (
    "The best time to visit Goa is November to February [S1]. "
    "Goa has beautiful beaches [S3]."
)

sources_2 = [
    {"source_id": "S1"},
    {"source_id": "S2"}
]


# Test 3: No citations
answer_3 = "Goa is a popular tourist destination."

sources_3 = [
    {"source_id": "S1"},
    {"source_id": "S2"}
]


for number, (answer, sources) in enumerate(
    [
        (answer_1, sources_1),
        (answer_2, sources_2),
        (answer_3, sources_3)
    ],
    start=1
):
    result = validate_citations(answer, sources)

    print(f"\n========== TEST {number} ==========")
    print("Citation validation:", result)