
EVALUATION_DATASET = [
    {
        "id": 1,
        "question": "What is the estimated budget for a 4-day Goa trip?",
        "expected_sources": ["Goa_4_Day_Trip.pdf"],
    },
    {
        "id": 2,
        "question": "What are the main attractions to visit in Goa?",
        "expected_sources": ["Goa_4_Day_Trip.pdf", "goa_guide.txt"],
    },
    {
        "id": 3,
        "question": "What food options are recommended in Goa?",
        "expected_sources": ["Goa_4_Day_Trip.pdf", "goa_guide.txt"],
    },
    {
        "id": 4,
        "question": "What are the main attractions in Meghalaya?",
        "expected_sources": [
            "Meghalaya_4_Day_Trip.pdf",
            "meghalaya_guide.txt",
        ],
    },
    {
        "id": 5,
        "question": "What is the estimated budget for a 4-day Thailand trip?",
        "expected_sources": ["Thailand_4_Day_Trip.pdf"],
    },

    {
        "id": 6,
        "question": "What is the population of Goa?",
        "expected_sources": [],
        "expected_behavior": "reject"
    },
    {
        "id": 7,
        "question": "Who is the current Chief Minister of Meghalaya?",
        "expected_sources": [],
        "expected_behavior": "reject"
    },
    {
        "id": 8,
        "question": "What is the weather forecast in Bangkok tomorrow?",
        "expected_sources": [],
        "expected_behavior": "reject"
    }

]
