# Vector Stores in LangChain

A comprehensive guide to vector stores, embeddings, similarity search,
retrievers, and their role in Retrieval-Augmented Generation (RAG).

## Table of Contents

1.  [What Is a Vector Store?](#1-what-is-a-vector-store)
2.  [Why Vector Stores Are Used in
    RAG](#2-why-vector-stores-are-used-in-rag)
3.  [Vectors and Embeddings](#3-vectors-and-embeddings)
4.  [What a Vector Store Contains](#4-what-a-vector-store-contains)
5.  [How Similarity Search Works](#5-how-similarity-search-works)
6.  [Similarity and Distance
    Metrics](#6-similarity-and-distance-metrics)
7.  [Popular Vector Stores and
    Libraries](#7-popular-vector-stores-and-libraries)
8.  [Using Vector Stores in
    LangChain](#8-using-vector-stores-in-langchain)
9.  [Important Vector Store Methods](#9-important-vector-store-methods)
10. [Vector Store vs. Retriever](#10-vector-store-vs-retriever)
11. [Search Strategies](#11-search-strategies)
12. [Metadata Filtering](#12-metadata-filtering)
13. [Persistence and Document IDs](#13-persistence-and-document-ids)
14. [Vector Stores vs. Traditional
    Databases](#14-vector-stores-vs-traditional-databases)
15. [Choosing a Chunking and Retrieval
    Strategy](#15-choosing-a-chunking-and-retrieval-strategy)
16. [Common Mistakes and Best
    Practices](#16-common-mistakes-and-best-practices)
17. [Complete RAG Workflow](#17-complete-rag-workflow)
18. [Quick Revision Notes](#18-quick-revision-notes)
19. [Practice Questions](#19-practice-questions)
20. [Useful Resources](#20-useful-resources)

------------------------------------------------------------------------

## 1. What Is a Vector Store?

A **Vector Store** is a system that stores numerical vectors
(embeddings) together with associated data, and provides ways to search
for vectors that are similar to a query vector.

In a RAG application, text is converted into embeddings by an embedding
model. The vectors are stored in a vector store. When a user asks a
question, the question is also embedded, and the vector store searches
for relevant stored vectors.

In simple terms:

-   **Embedding model:** Converts text into numerical vectors.
-   **Vector store:** Stores vectors and associated documents or
    references.
-   **Similarity search:** Finds vectors that are close to a query
    vector.
-   **Retriever:** Provides a standard interface for fetching relevant
    documents.

A vector is an ordered list of numbers:

``` python
vector = [0.21, 0.87, -0.32, 0.45]
```

This is only an illustrative vector. Real embedding models commonly
produce vectors with hundreds or thousands of dimensions.

## 2. Why Vector Stores Are Used in RAG

A RAG application needs to find relevant information from a collection
of documents before asking an LLM to answer a question.

A typical ingestion and retrieval workflow is:

``` text
                 DOCUMENT INGESTION
Documents
    |
Document Loaders
    |
Text Splitters
    |
Document Chunks
    |
Embedding Model
    |
Vector Store
    |
    +----------------------------------+
                                       |
                 QUERY / RETRIEVAL     |
                                       |
User Question                          |
    |                                  |
Embedding Model                        |
    |                                  |
Query Vector                           |
    |                                  |
Vector Similarity Search --------------+
    |
Relevant Documents
    |
Prompt + Retrieved Context
    |
LLM
    |
Final Answer
```

Why not pass an entire large document to the LLM?

-   Large documents may exceed model context limits.
-   Sending unnecessary content increases token usage and latency.
-   A focused retrieval step can provide more relevant context.
-   Source metadata can help identify where retrieved information came
    from.

A vector store is one common way to implement the retrieval layer. RAG
systems can also use keyword search, relational databases with vector
extensions, or hybrid retrieval.

## 3. Vectors and Embeddings

An **embedding** is a numerical representation of data, such as text,
designed so that useful relationships between items can be measured in
vector space.

For example, an embedding model may map these sentences to nearby
vectors:

-   "How can I learn artificial intelligence?"
-   "What is a good way to study AI?"

The wording differs, but the meanings are related. A suitable embedding
model may represent them as vectors that are close in its embedding
space.

Example:

``` python
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector = embedding_model.embed_query(
    "What is machine learning?"
)

print(len(vector))
print(vector[:5])
```

The `all-MiniLM-L6-v2` model produces 384-dimensional embeddings.

### `embed_query()` vs. `embed_documents()`

Many LangChain embedding integrations expose two methods:

``` python
query_vector = embedding_model.embed_query("What is RAG?")

document_vectors = embedding_model.embed_documents([
    "RAG combines retrieval with generation.",
    "Embeddings represent data numerically."
])
```

-   `embed_query()` embeds a search query.
-   `embed_documents()` embeds a list of documents for indexing.

Some embedding models use different query and document instructions or
prefixes. Use the embedding integration's recommended methods rather
than manually assuming both paths are identical.

## 4. What a Vector Store Contains

A vector record commonly associates three kinds of information:

  Field      Meaning
  ---------- -------------------------------------
  ID         Unique identifier for the record
  Vector     Numerical embedding
  Document   Original text or a reference to it
  Metadata   Extra attributes about the document

Example conceptual record:

``` python
{
    "id": "doc_001",
    "vector": [0.21, 0.87, -0.32, 0.45],
    "text": "Machine Learning is a subset of AI.",
    "metadata": {
        "source": "ml_notes.pdf",
        "page": 5,
        "topic": "Machine Learning"
    }
}
```

The exact internal storage format depends on the vector store. Some
systems store the original text; others can store references or external
payloads.

### Why metadata matters

Metadata makes it possible to:

-   Identify the source file.
-   Track page numbers.
-   Filter by topic, author, date, or category.
-   Display citations or source references.
-   Restrict retrieval to a particular collection or document set.

## 5. How Similarity Search Works

Suppose a vector store contains these documents:

1.  Machine Learning is a subset of Artificial Intelligence.
2.  Deep Learning uses neural networks.
3.  Python is a programming language.

The user asks:

> What is a subset of AI?

The retrieval process is:

1.  Convert the user's question into a query embedding.
2.  Compare the query vector with indexed document vectors.
3.  Rank candidates using the configured similarity or distance measure.
4.  Return the top matching documents.

The exact results depend on the embedding model, metric, index, filters,
and search configuration.

### Top-K retrieval

`k` specifies how many results to return.

``` python
results = vector_store.similarity_search(
    "What is a subset of AI?",
    k=2
)
```

Here, the vector store returns up to two matching documents.

A larger `k` can provide more context, but may also introduce irrelevant
information. Choose `k` through retrieval evaluation rather than
assuming a larger value is always better.

## 6. Similarity and Distance Metrics

Vector stores can use different mathematical measures to compare
vectors. The metric must be appropriate for the embedding model and
index configuration.

### A. Cosine Similarity

Cosine similarity measures the cosine of the angle between two vectors:

\[ `\operatorname{cosine}`{=tex}(A,B)=
`\frac{A\cdot B}{\|A\|\|B\|}`{=tex} \]

Where:

-   \(A\) and (B) are vectors.
-   (A `\cdot `{=tex}B) is their dot product.
-   (\|A\|) and (\|B\|) are their magnitudes.

For non-zero vectors:

-   `1` means the vectors point in the same direction.
-   `0` means they are perpendicular.
-   `-1` means they point in opposite directions.

Cosine similarity measures direction and is not directly affected by
vector magnitude.

### B. Euclidean Distance (L2)

Euclidean distance measures straight-line distance between two vectors:

\[ d(A,B)=`\sqrt{\sum_{i=1}^{n}(A_i-B_i)^2}`{=tex} \]

A smaller distance means the vectors are closer under this metric.

### C. Dot Product

The dot product is:

\[ A`\cdot `{=tex}B=`\sum`{=tex}\_{i=1}\^{n}A_iB_i \]

A larger dot product can indicate greater alignment, but vector
magnitudes also affect the result.

### Important metric distinction

Do not assume every vector store returns cosine similarity. Some
integrations return distances, where **lower is better**. Others return
similarity scores, where **higher is better**. Read the integration's
documentation to understand the score semantics.

For example, Chroma's `similarity_search_with_score()` returns a
distance score; a lower distance indicates a closer match.

## 7. Popular Vector Stores and Libraries

### Chroma

-   Open-source vector database.
-   Straightforward local development workflow.
-   Useful for learning, prototypes, and smaller RAG applications.
-   Supports persistence and metadata filtering.

Website: https://www.trychroma.com/

### FAISS

-   Vector similarity-search library developed by Meta.
-   Efficient indexing and search over dense vectors.
-   Useful for local experimentation and large-scale similarity-search
    workloads.
-   It is a vector search library, not a full database service with all
    the features of a managed database.

Website: https://faiss.ai/

### Pinecone

-   Managed vector database platform.
-   Useful when a hosted vector database is preferred.
-   Provides production-oriented indexing and retrieval capabilities.

Website: https://www.pinecone.io/

### Weaviate

-   Open-source vector database with managed deployment options.
-   Supports vector search, metadata filtering, and hybrid search.

Website: https://weaviate.io/

### Milvus

-   Open-source vector database.
-   Designed for large-scale vector search and distributed deployments.

Website: https://milvus.io/

### PostgreSQL with pgvector

-   Adds vector similarity search capabilities to PostgreSQL.
-   Useful when an application already uses PostgreSQL and wants
    relational data and vector search together.

Website: https://github.com/pgvector/pgvector

### Which one should you learn first?

For a beginner building a local RAG project:

1.  Start with Chroma to understand the concepts.
2.  Learn FAISS to understand vector indexing and similarity search.
3.  Explore a managed service such as Pinecone when you need hosted
    infrastructure.
4.  Explore pgvector if you want to combine relational data and vector
    search in PostgreSQL.

The best choice depends on deployment requirements, scale, filtering
needs, operational constraints, and budget.

## 8. Using Vector Stores in LangChain

The following example uses Chroma and Hugging Face embeddings.

### Step 1: Install dependencies

``` bash
pip install langchain-core langchain-chroma langchain-huggingface sentence-transformers
```

If you are loading documents from files, install the relevant loader
integration too. For example:

``` bash
pip install langchain-community
```

### Step 2: Initialize an embedding model

``` python
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

The first run may download the model files. This model is suitable for
experimentation, but evaluate embedding quality for your actual use
case.

### Step 3: Prepare documents

LangChain represents content using `Document` objects.

``` python
from langchain_core.documents import Document

documents = [
    Document(
        page_content="Machine Learning is a subset of AI.",
        metadata={"topic": "ML", "source": "notes.txt"}
    ),
    Document(
        page_content="Deep Learning uses neural networks.",
        metadata={"topic": "DL", "source": "notes.txt"}
    ),
    Document(
        page_content="NLP enables computers to process human language.",
        metadata={"topic": "NLP", "source": "notes.txt"}
    )
]
```

A `Document` typically has:

-   `page_content`: The text.
-   `metadata`: A dictionary of additional information.

### Step 4: Create a Chroma vector store

``` python
from langchain_chroma import Chroma

vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_name="my_collection",
    persist_directory="./chroma_db"
)
```

Conceptually, LangChain:

1.  Receives the documents.
2.  Uses the embedding model to embed them.
3.  Stores the vectors and associated document information in Chroma.
4.  Persists the collection in the specified directory.

Chroma's local persistence behavior is managed by the integration; do
not assume every vector store has the same persistence API.

### Step 5: Perform similarity search

``` python
query = "What is a subset of AI?"

results = vector_store.similarity_search(
    query,
    k=2
)

for doc in results:
    print("Content:", doc.page_content)
    print("Metadata:", doc.metadata)
    print("-" * 50)
```

### Step 6: Similarity search with scores

``` python
results = vector_store.similarity_search_with_score(
    "What is a subset of AI?",
    k=2
)

for doc, score in results:
    print("Content:", doc.page_content)
    print("Distance:", score)
    print("Metadata:", doc.metadata)
```

For this Chroma example, the score is a distance, not a
cosine-similarity value. Lower distance means closer vectors.

### Step 7: Add more documents

``` python
new_documents = [
    Document(
        page_content="Transformers use attention mechanisms.",
        metadata={"topic": "Transformers"}
    )
]

vector_store.add_documents(new_documents)
```

For reliable updates and deletion, use stable document IDs where
supported and retain those IDs in your application.

### Step 8: Convert the vector store into a retriever

``` python
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

results = retriever.invoke("Explain deep learning")

for doc in results:
    print(doc.page_content)
```

The retriever provides a standardized interface for fetching documents.
In LangChain, retrievers implement the Runnable interface, so they can
be composed into LCEL chains.

## 9. Important Vector Store Methods

Exact method availability and arguments can vary by integration. These
are common LangChain vector store operations.

  Method                             Purpose
  ---------------------------------- ------------------------------------------------
  `from_documents()`                 Create a vector store from LangChain documents
  `from_texts()`                     Create a vector store from raw text
  `add_documents()`                  Add documents to an existing store
  `add_texts()`                      Add raw text
  `similarity_search()`              Retrieve similar documents
  `similarity_search_with_score()`   Retrieve documents with a score
  `delete()`                         Delete records, often by IDs
  `as_retriever()`                   Create a Retriever interface

Example:

``` python
# Add documents
vector_store.add_documents(documents)

# Search
results = vector_store.similarity_search(
    "What is NLP?",
    k=3
)

# Convert to retriever
retriever = vector_store.as_retriever()
```

Check the specific integration documentation for supported arguments and
deletion behavior.

## 10. Vector Store vs. Retriever

These concepts are related but not identical.

