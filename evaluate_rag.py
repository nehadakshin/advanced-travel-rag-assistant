
import time
import json
from pathlib import Path

from src.vector_store import load_vector_store
from src.retrieval import answer_question
from src.evaluation import evaluate_answer
from evaluation_dataset import EVALUATION_DATASET


def is_clear_refusal(answer):
    """
    Check whether the answer clearly indicates
    that the uploaded documents lack the information.
    """
    refusal_phrases = [
        "couldn't find sufficiently relevant information",
        "not enough information",
        "do not contain enough information",
        "does not contain enough information",
        "cannot answer based on the uploaded documents",
        "not available in the uploaded documents",
        "unable to answer based on the provided context"
    ]

    answer_lower = answer.lower()

    return any(
        phrase in answer_lower
        for phrase in refusal_phrases
    )


def evaluate_rag():

    print("\n========== RAG + LLM JUDGE EVALUATION ==========\n")

    vector_store = load_vector_store()

    results = []

    total_tests = len(EVALUATION_DATASET)

    # Answerable-question metrics
    answerable_tests = 0
    retrieval_hits = 0
    total_source_coverage = 0
    total_judge_score = 0

    # Rejection metrics
    rejection_tests = 0
    correct_rejections = 0

    # Overall metrics
    total_latency = 0
    correct_predictions = 0

    for test_case in EVALUATION_DATASET:

        question = test_case["question"]
        expected_sources = set(test_case["expected_sources"])

        expected_behavior = test_case.get(
            "expected_behavior", "answer"
        )

        print(f"\nTest Case {test_case['id']}")
        print(f"Question: {question}")
        print(f"Expected behavior: {expected_behavior}")

        start_time = time.perf_counter()

        # Run the actual RAG pipeline
        response = answer_question(
            vector_store=vector_store,
            question=question,
            k=4
        )

        rag_latency = time.perf_counter() - start_time
        total_latency += rag_latency

        # Extract retrieved sources
        retrieved_sources = {
            Path(source["source"]).name
            for source in response["sources"]
        }

        # Prepare common result fields
        result = {
            "test_id": test_case["id"],
            "question": question,
            "expected_behavior": expected_behavior,
            "expected_sources": sorted(expected_sources),
            "retrieved_sources": sorted(retrieved_sources),
            "rag_latency_seconds": round(rag_latency, 2),
            "answer": response["answer"],
            "citation_validation": response["citation_validation"]
        }

        print(f"\nRetrieved sources: {sorted(retrieved_sources)}")
        print(f"RAG latency: {rag_latency:.2f} seconds")

        # -----------------------------------------
        # CASE 1: EXPECTED TO ANSWER
        # -----------------------------------------

        if expected_behavior == "answer":

            answerable_tests += 1

            matched_sources = expected_sources.intersection(
                retrieved_sources
            )

            retrieval_hit = bool(matched_sources)

            if retrieval_hit:
                retrieval_hits += 1
                correct_predictions += 1

            source_coverage = (
                len(matched_sources) / len(expected_sources)
                if expected_sources else 0
            )

            total_source_coverage += source_coverage

            # Evaluate answer using Gemini LLM judge
            valid_source_ids = [
                source["source_id"]
                for source in response["sources"]
            ]

            judge_result = evaluate_answer(
                question=question,
                context=response["context"],
                answer=response["answer"],
                valid_source_ids=valid_source_ids
            )

            total_judge_score += judge_result["overall_score"]

            result.update({
                "retrieval_hit": retrieval_hit,
                "source_coverage": round(source_coverage, 2),
                "llm_judge": judge_result
            })

            print(f"Retrieval hit: {retrieval_hit}")
            print(f"Source coverage: {source_coverage:.2%}")

            print("\nLLM Judge Scores:")

            for metric in [
                "faithfulness",
                "relevance",
                "completeness",
                "citation_accuracy"
            ]:
                score = judge_result[metric]["score"]
                reason = judge_result[metric]["reason"]

                print(f"{metric}: {score}/5")
                print(f"Reason: {reason}")

            print(
                f"\nOverall judge score: "
                f"{judge_result['overall_score']}/5"
            )

        # -----------------------------------------
        # CASE 2: EXPECTED TO REJECT
        # -----------------------------------------

        elif expected_behavior == "reject":

            rejection_tests += 1

            # A correct rejection requires:
            # 1. No documents were retrieved.
            # 2. The answer clearly indicates insufficient information.

            no_relevant_sources = len(response["sources"]) == 0

            clear_refusal = is_clear_refusal(
                response["answer"]
            )

            correct_rejection = (
                no_relevant_sources and clear_refusal
            )

            if correct_rejection:
                correct_rejections += 1
                correct_predictions += 1

            result.update({
                "no_relevant_sources": no_relevant_sources,
                "clear_refusal": clear_refusal,
                "correct_rejection": correct_rejection
            })

            print(f"No relevant sources: {no_relevant_sources}")
            print(f"Clear refusal: {clear_refusal}")
            print(f"Correct rejection: {correct_rejection}")

        else:
            raise ValueError(
                f"Unknown expected_behavior: {expected_behavior}"
            )

        results.append(result)

        print("\n" + "-" * 60)

    # -----------------------------------------
    # AGGREGATE METRICS
    # -----------------------------------------

    retrieval_accuracy = (
        retrieval_hits / answerable_tests
        if answerable_tests else 0
    )

    average_source_coverage = (
        total_source_coverage / answerable_tests
        if answerable_tests else 0
    )

    rejection_accuracy = (
        correct_rejections / rejection_tests
        if rejection_tests else 0
    )

    overall_accuracy = (
        correct_predictions / total_tests
        if total_tests else 0
    )

    average_latency = (
        total_latency / total_tests
        if total_tests else 0
    )

    average_judge_score = (
        total_judge_score / answerable_tests
        if answerable_tests else 0
    )

    summary = {
        "total_test_cases": total_tests,
        "answerable_test_cases": answerable_tests,
        "rejection_test_cases": rejection_tests,

        "retrieval_accuracy": round(retrieval_accuracy, 2),
        "average_source_coverage": round(
            average_source_coverage, 2
        ),

        "rejection_accuracy": round(rejection_accuracy, 2),
        "overall_task_accuracy": round(overall_accuracy, 2),

        "average_rag_latency_seconds": round(
            average_latency, 2
        ),

        "average_llm_judge_score": round(
            average_judge_score, 2
        ),

        "individual_results": results
    }

    # -----------------------------------------
    # DISPLAY SUMMARY
    # -----------------------------------------

    print("\n========== FINAL EVALUATION SUMMARY ==========")

    print(f"Total test cases: {total_tests}")
    print(f"Answerable questions: {answerable_tests}")
    print(f"Rejection questions: {rejection_tests}")

    print(f"\nRetrieval accuracy: {retrieval_accuracy:.2%}")
    print(f"Average source coverage: {average_source_coverage:.2%}")

    print(f"Rejection accuracy: {rejection_accuracy:.2%}")
    print(f"Overall task accuracy: {overall_accuracy:.2%}")

    print(f"Average RAG latency: {average_latency:.2f} seconds")
    print(f"Average LLM judge score: {average_judge_score:.2f}/5")

    # Save results
    output_path = Path("evaluation_results_llm_judge.json")

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    evaluate_rag()
