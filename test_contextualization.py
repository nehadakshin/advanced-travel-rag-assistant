from src.retrieval import contextualize_question


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

result = contextualize_question(
    question,
    chat_history
)

print("\n========== CONTEXTUALIZATION TEST ==========")
print("Original question:", question)
print("Reformulated question:", result)