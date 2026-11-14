# LangChain Document Loaders

This folder contains small examples of loading web pages, text files, PDFs, and a directory of PDFs into LangChain `Document` objects. These loaders are useful at the start of a retrieval-augmented generation (RAG) workflow: they extract text and attach metadata that later steps can split, embed, index, or send to a model.

## Examples

| File | What it demonstrates |
| --- | --- |
| `text_loader.py` | Loads `RAG\Docs\black_holes.txt` with `TextLoader`, then sends its text to an Ollama chat model to generate a summary. |
| `PyPDF_loader.py` | Loads `RAG\Docs\black_holes.pdf` with `PyPDFLoader` and summarizes the first page with an Ollama chat model. |
| `Directory_loader.py` | Uses `DirectoryLoader` with `PyPDFLoader` to load PDFs from `RAG\Books`. It shows both eager loading with `load()` and on-demand iteration with `lazy_load()`. |
| `WebBaseLoader.py` | Loads a product page using `WebBaseLoader` and prints the number of returned documents and the page text. |
| `Product_review.py` | Experiments with loading a product page and passing its content to an Ollama model to answer a question. |

## What a loader returns

Most examples return a list of LangChain `Document` objects. A document has:

- `page_content`: the extracted text.
- `metadata`: source information, such as a file path, page number, or URL, depending on the loader.

For example, `docs[0].page_content` accesses the first document's text, while `docs[0].metadata` accesses its metadata. A PDF loader may return one document per page; a directory loader can return documents from multiple files.

## Running the examples

Run commands from the repository root because the file examples use paths such as `RAG\Docs\black_holes.txt`:

```powershell
python RAG\DocLoaders\text_loader.py
python RAG\DocLoaders\PyPDF_loader.py
python RAG\DocLoaders\Directory_loader.py
python RAG\DocLoaders\WebBaseLoader.py
python RAG\DocLoaders\Product_review.py
```

The examples use packages that are not all listed in the repository's `requirements.txt`. Depending on which scripts you run, install the corresponding integrations and community loaders, such as `langchain-community`, `langchain-ollama`, and `pypdf`. The summarization examples also require Ollama to be installed and running with the `llama3.2:3b` model available.

## Notes

- `Directory_loader.py` prints `docs[100]`, which will raise an `IndexError` unless at least 101 page documents were loaded. Change the index or remove that print if the PDFs produce fewer documents.
- The web examples use hard-coded Amazon URLs. Pages may change, require network access, or block automated requests.
- `Product_review.py` is an experiment rather than a guaranteed runnable example: its prompt declares `questions` while the template and invocation use `question`, and it passes the loader's result instead of extracted page text. Align the prompt variable name and pass the desired document's `page_content` before running it.
- The examples demonstrate loading and basic model calls; they do not yet split documents or build a vector store and retriever.
