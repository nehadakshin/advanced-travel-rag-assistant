# advanced-travel-rag-assistant

# Advanced Travel Knowledge Assistant — RAG

An AI-powered Travel Knowledge Assistant built using **Retrieval-Augmented Generation (RAG)**, Google Gemini, LangChain, and FAISS.

The application allows users to upload travel documents in PDF and TXT formats and ask natural-language questions about destinations, budgets, attractions, food, transportation, and travel plans.

Unlike a basic document chatbot, this project incorporates contextual retrieval, similarity filtering, citation validation, incremental indexing, failure recovery, and LLM-based evaluation to improve reliability.

## Features

- PDF and TXT document ingestion.
- Text extraction and metadata preservation.
- Recursive text chunking with configurable overlap.
- Gemini-based document and query embeddings.
- FAISS vector database for semantic search.
- Similarity threshold filtering.
- Context-aware question rewriting for follow-up questions.
- Source-grounded answer generation.
- Source citations in generated answers.
- Citation ID validation.
- Chat history support.
- Incremental document indexing.
- Detection of new, modified, unchanged, and deleted documents.
- Staged indexing and rollback on commit failures.
- Streamlit interface for document uploads and question answering.
- Retrieval evaluation and LLM-as-a-judge evaluation.
- Out-of-scope question rejection.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application development |
| LangChain | RAG pipeline and LLM integration |
| Google Gemini | Answer generation and evaluation |
| Gemini Embedding API | Text embeddings |
| FAISS | Vector storage and similarity search |
| PyPDFLoader | PDF document extraction |
| TextLoader | TXT document extraction |
| RecursiveCharacterTextSplitter | Text chunking |
| Streamlit | User interface |
| python-dotenv | API key management |

## Architecture

```text
                User
                  |
                  v
          Streamlit Interface
                  |
                  v
           User Question
                  |
                  v
       Question Contextualization
                  |
                  v
          Gemini Embeddings
                  |
                  v
          FAISS Vector Search
                  |
                  v
       Similarity Threshold Filter
                  |
           +------+------+
           |             |
       Relevant       No Relevant
        Chunks          Chunks
           |             |
           v             v
      Context Build   Refusal Response
           |
           v
      Gemini LLM
           |
           v
    Citation Validation
           |
           v
     Answer + Sources
```

### Document Indexing Pipeline

```text
PDF / TXT Documents
        |
        v
   Document Loading
        |
        v
    Text Extraction
        |
        v
    Text Chunking
        |
        v
 Gemini Embeddings
        |
        v
    FAISS Index
        |
        v
 Local Persistent Storage
```

## Key RAG Implementation Techniques

### 1. Document Ingestion

Implemented document loading for both PDF and TXT files using LangChain loaders.

- `PyPDFLoader` for PDF documents.
- `TextLoader` for text documents.
- Metadata preservation to identify the original document.
- Selective document loading through `load_selected_documents()`.

This provides a common ingestion pipeline for different document formats.

### 2. Recursive Text Chunking

Used `RecursiveCharacterTextSplitter` to divide large documents into smaller, overlapping chunks.

Configuration:

```python
chunk_size = 500
chunk_overlap = 50

separators = [
    "\n\n",
    "\n",
    " ",
    ""
]
```

Additional metadata:
- `start_index`
- `chunk_index`
- Original source filename

**Why this matters:**

Chunking prevents entire documents from being passed to the LLM. Overlapping chunks also help preserve context across chunk boundaries.

### 3. Gemini Embeddings

Used Google's `gemini-embedding-001` embedding model to convert document chunks and user queries into vector representations.

Implemented separate embedding task types:

- `RETRIEVAL_DOCUMENT` for document indexing.
- `RETRIEVAL_QUERY` for user queries.

Created a custom `GeminiRetrievalEmbeddings` wrapper to integrate the embedding API with the LangChain vector store.

**Why this matters:**

Semantic embeddings allow the system to retrieve information based on meaning rather than relying only on exact keyword matches.

### 4. FAISS Vector Database

Used FAISS for storing and retrieving document embeddings.

Implemented:
- Vector index creation.
- Similarity search.
- Persistent local storage.
- Loading existing indexes.
- Document deletion and updates.

Used normalized L2 distance for similarity retrieval.

**Why this matters:**

FAISS provides efficient vector similarity search and allows the application to reuse an existing index instead of embedding every document again.

### 5. Similarity Threshold Filtering

Implemented a configurable distance threshold to reject irrelevant retrieval results.

```python
def retrieve_documents(
    vector_store,
    question,
    k=4,
    score_threshold=0.60
):
    results = vector_store.similarity_search_with_score(
        question,
        k=k
    )

    accepted = [
        doc
        for doc, score in results
        if score <= score_threshold
    ]

    return accepted
```

The threshold is dataset-specific and should be calibrated against evaluation data.

**Why this matters:**

Without relevance filtering, the LLM might generate answers using unrelated document chunks. Filtering helps the system abstain when the available context is insufficient.

### 6. Contextual Question Rewriting

Implemented `contextualize_question()` to convert follow-up questions into standalone questions using previous conversation history.

Example:

```text
Previous question:
What is the estimated budget for a Goa trip?

Follow-up:
What about the food?

Rewritten question:
What are the estimated food expenses
for a Goa trip?
```

The rewritten question is used for retrieval.

**Why this matters:**

Vector retrieval should understand the actual subject of a follow-up question. This avoids losing important context from earlier conversation turns.

### 7. Context Assembly and Grounded Generation

Implemented `build_context()` to combine retrieved document chunks and assign source identifiers.

Example:

```text
[S1] Goa travel budget information...

[S2] Goa attractions and sightseeing information...
```

The Gemini prompt instructs the model to:

- Answer using only the supplied context.
- Avoid inventing facts.
- Clearly indicate when information is unavailable.
- Include source citations such as `[S1]` and `[S2]`.
- Treat document content as reference material, not as instructions.

**Why this matters:**

Grounded generation reduces unsupported answers and makes the output traceable to the original documents.

### 8. Citation Validation

Implemented `validate_citations()` to check whether citation IDs appearing in the generated answer exist in the retrieved source list.

Example:

```text
Generated answer:
The estimated budget is ₹15,000–₹20,000 [S1].

Available source IDs:
[S1], [S2]

Citation validation:
Valid
```

This validates citation identifier membership. It does not independently verify that every cited statement is semantically supported by its source.

### 9. Incremental Document Indexing

Implemented incremental indexing to avoid rebuilding the entire vector database whenever documents change.

The document tracker uses file hashes to identify:

- New documents.
- Modified documents.
- Unchanged documents.
- Deleted documents.

Only changed documents need to be processed again.

Implemented components include:

- `calculate_file_hash()`
- `get_document_hashes()`
- `detect_document_changes()`
- `run_incremental_indexing()`

**Why this matters:**

Incremental indexing reduces unnecessary embedding API calls and makes document updates more efficient.

### 10. Staging and Rollback for Index Updates

Implemented a staged update approach for safer index replacement.

The update process:

1. Detect document changes.
2. Prepare the updated index in a staging location.
3. Prepare the updated document manifest.
4. Back up existing index and manifest.
5. Replace the existing files with staged versions.
6. Restore backups if a commit operation fails.

This improves recovery from ordinary update failures. It is not a fully crash-proof transaction across multiple directories.

### 11. Load-or-Create Vector Store

Implemented logic to check whether a persisted FAISS index already exists.

- If an index exists, load it.
- If an index does not exist, create it from the documents.
- Avoid unnecessary document embedding on subsequent application starts.

This improves application startup and avoids repeated embedding operations.

### 12. Out-of-Scope Question Rejection

Implemented a retrieval-based rejection mechanism.

Example questions:

- What is the population of Goa?
- Who is the current Chief Minister of Meghalaya?
- What is the weather forecast in Bangkok tomorrow?

When no relevant chunks pass the similarity threshold, the application returns an abstention response instead of generating an unsupported answer.

Example:

```text
I couldn't find sufficiently relevant information
in the uploaded documents.
```

This is particularly important for reducing hallucinations in knowledge-grounded applications.

## Evaluation Framework

Implemented an evaluation pipeline using:

- Retrieval metrics.
- Expected-source coverage.
- Response latency.
- Citation validation.
- Gemini as an LLM judge.
- Out-of-scope rejection accuracy.

### LLM-as-a-Judge Evaluation

The evaluator scores generated answers across four dimensions:

| Dimension | What it evaluates |
|---|---|
| Faithfulness | Whether claims are supported by retrieved context |
| Relevance | Whether the answer addresses the question |
| Completeness | Whether important information is included |
| Citation Accuracy | Whether citations refer to valid and supporting sources |

Each dimension receives a score from 1 to 5.

### Evaluation Results

Initial evaluation using eight test cases:

| Metric | Result |
|---|---:|
| Total test cases | 8 |
| Answerable questions | 5 |
| Out-of-scope questions | 3 |
| Retrieval accuracy | 100% |
| Average source coverage | 80% |
| Rejection accuracy | 100% |
| Overall task accuracy | 100% |
| Average RAG latency | 1.28 seconds |
| Average LLM judge score | 5.0/5 |

**Note:** These are results from a small prototype evaluation dataset, not a production benchmark. LLM judge scores are model-generated assessments and require independent human validation.

## Project Structure

```text
travel-rag/
│
├── app.py
├── evaluate_rag.py
├── evaluation_dataset.py
├── requirements.txt
├── .env
├── .gitignore
│
├── documents/
│   ├── Goa_4_Day_Trip.pdf
│   ├── Meghalaya_4_Day_Trip.pdf
│   ├── Thailand_4_Day_Trip.pdf
│   ├── goa_guide.txt
│   └── meghalaya_guide.txt
│
├── vector_db/
│   └── travel_faiss/
│
├── metadata/
│   └── document_manifest.json
│
└── src/
    ├── ingestion.py
    ├── chunking.py
    ├── embeddings.py
    ├── vector_store.py
    ├── retrieval.py
    ├── document_tracker.py
    ├── incremental_indexer.py
    └── evaluation.py
```

