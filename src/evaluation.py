
import json
import re

from langchain_google_genai import ChatGoogleGenerativeAI


evaluator = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0,
    max_retries=2,
    timeout=60,
)


def evaluate_answer(question, context, answer, valid_source_ids):
    """
    Evaluate a RAG-generated answer using Gemini.

    Scores:
        Faithfulness: 1-5
        Relevance: 1-5
        Completeness: 1-5
        Citation accuracy: 1-5
    """

    prompt = f"""
You are an expert evaluator assessing a Retrieval-Augmented
Generation (RAG) system.

Evaluate the answer using ONLY the supplied question and context.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

VALID SOURCE IDS:
{valid_source_ids}

GENERATED ANSWER:
{answer}

Evaluate the following dimensions:

1. faithfulness:
Score whether factual claims in the answer are supported
by the retrieved context.

2. relevance:
Score whether the answer directly addresses the question.

3. completeness:
Score whether the answer includes the important information
available in the retrieved context.

4. citation_accuracy:
Score whether citations refer to valid source IDs and whether
the cited context supports the associated claims.
A citation that merely exists is not automatically accurate.

Scoring:
1 = Very poor
2 = Poor
3 = Partially satisfactory
4 = Good
5 = Excellent

Rules:
- Do not use external knowledge.
- Do not assume unsupported facts are correct.
- Do not reward an answer for including information absent
  from the retrieved context.
- If no citations are provided, citation accuracy should be 1.
- Return valid JSON only.
- Include a short explanation for each score.

Return exactly this structure:

{{
    "faithfulness": {{
        "score": 1,
        "reason": "Explanation"
    }},
    "relevance": {{
        "score": 1,
        "reason": "Explanation"
    }},
    "completeness": {{
        "score": 1,
        "reason": "Explanation"
    }},
    "citation_accuracy": {{
        "score": 1,
        "reason": "Explanation"
    }},
    "overall_score": 1.0
}}
"""

    response = evaluator.invoke(prompt)

    content = response.content

    if isinstance(content, list):
        content = "\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict)
        )

    content = str(content).strip()

    # Remove Markdown JSON fences if Gemini returns them.
    content = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        content,
        flags=re.IGNORECASE,
    ).strip()

    result = json.loads(content)

    # Validate scores before accepting the evaluation.
    dimensions = [
        "faithfulness",
        "relevance",
        "completeness",
        "citation_accuracy",
    ]

    scores = []

    for dimension in dimensions:
        score = result[dimension]["score"]

        if not isinstance(score, (int, float)) or not 1 <= score <= 5:
            raise ValueError(
                f"Invalid score for {dimension}: {score}"
            )

        scores.append(score)

    result["overall_score"] = round(
        sum(scores) / len(scores),
        2,
    )

    return result
