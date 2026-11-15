
from pathlib import Path

import streamlit as st

from src.retrieval import answer_question
from src.vector_store import load_vector_store
from src.incremental_indexer import run_incremental_indexing


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DOCUMENTS_DIR = Path("documents")
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="Travel Knowledge Assistant",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Travel Knowledge Assistant")
st.write("Upload travel documents and explore their contents.")


# --------------------------------------------------
# STEP 1: INITIALIZE SESSION STATE
# --------------------------------------------------

if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = None


# --------------------------------------------------
# STEP 2: LOAD EXISTING FAISS INDEX
# --------------------------------------------------

if st.session_state["vector_store"] is None:

    try:
        vector_store = load_vector_store()

        st.session_state["vector_store"] = vector_store

        st.success(
            f"FAISS database loaded successfully! "
            f"Total vectors: {vector_store.index.ntotal}"
        )

    except FileNotFoundError:
        st.info(
            "No existing FAISS database found. "
            "Please upload documents and update the knowledge base."
        )

    except Exception as error:
        st.warning(f"Unable to load FAISS database: {error}")


# --------------------------------------------------
# STEP 3: UPLOAD AND INCREMENTALLY UPDATE DOCUMENTS
# --------------------------------------------------

st.divider()

st.subheader("1. Upload Travel Documents")

uploaded_files = st.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if st.button("Update Knowledge Base"):

    try:

        # Save uploaded documents
        if uploaded_files:

            for uploaded_file in uploaded_files:

                file_path = DOCUMENTS_DIR / uploaded_file.name

                with open(file_path, "wb") as file:
                    file.write(uploaded_file.getbuffer())

                st.write(f"Saved: {uploaded_file.name}")

        # Run incremental indexing
        with st.spinner(
            "Detecting document changes and updating FAISS..."
        ):

            result = run_incremental_indexing(
                documents_directory=str(DOCUMENTS_DIR)
            )

        vector_store = result["vector_store"]
        stats = result["stats"]

        # Update session state only after successful indexing
        st.session_state["vector_store"] = vector_store

        st.success("Knowledge base updated successfully!")

        # Display indexing statistics
        st.subheader("Indexing Summary")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("New Documents", stats["new_documents"])
        col2.metric("Modified", stats["modified_documents"])
        col3.metric("Deleted", stats["deleted_documents"])
        col4.metric("Unchanged", stats["unchanged_documents"])

        col5, col6, col7 = st.columns(3)

        col5.metric("Chunks Added", stats["added_chunks"])
        col6.metric("Vectors Deleted", stats["deleted_vectors"])
        col7.metric("Total Vectors", stats["total_vectors"])

    except Exception as error:

        st.error(f"Knowledge base update failed: {error}")

        st.warning(
            "The previous production index should remain available. "
            "Resolve the error and try again."
        )


# --------------------------------------------------
# STEP 4: ASK QUESTIONS
# --------------------------------------------------

st.divider()

st.subheader("2. Ask Your Travel Knowledge Assistant")

question = st.text_area(
    "Ask a question about your travel documents",
    placeholder="Example: Plan a 4-day Goa trip."
)

if st.button("Generate Answer"):

    vector_store = st.session_state.get("vector_store")

    if vector_store is None:

        st.warning(
            "Please upload documents and update the knowledge base first."
        )

    elif not question.strip():

        st.warning("Please enter a question.")

    else:

        try:

            with st.spinner(
                "Searching documents and generating your answer..."
            ):

                result = answer_question(
                    vector_store=vector_store,
                    question=question,
                    k=4
                )

            st.subheader("AI Generated Answer")

            st.markdown(result["answer"])

            # Display retrieved sources
            st.subheader("Sources")

            if result["sources"]:

                for source in result["sources"]:

                    with st.expander(
                        f"{source['source_id']} - {source['source']}"
                    ):

                        st.write(
                            "Chunk:",
                            source["chunk_index"]
                        )

                        st.write("Retrieved content:")

                        st.write(source["content"])

            else:

                st.info("No source documents were retrieved.")

        except Exception as error:

            st.error(f"Unable to generate answer: {error}")
