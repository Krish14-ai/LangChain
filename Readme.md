# LangChain Learning Examples

This repository is a collection of small, standalone Python examples for learning LangChain concepts. It is organized by topic rather than as one application; run the example you want to explore from the repository root.

## What’s included

| Folder | Topics |
| --- | --- |
| `Chains/` | Sequential, parallel, multi-step, and conditional chains |
| `ChatModels/` | Chat model integrations for Google Gemini, Anthropic, and Hugging Face |
| `EmbeddingModels/` | Google and Hugging Face embeddings, document embeddings, and cosine-similarity search |
| `messeges/` | Message history and `MessagesPlaceholder` examples (the folder name is intentionally kept as it is in the repository) |
| `outputs/` | Structured model output using JSON schemas and Pydantic, plus string, JSON, and Pydantic output parsers |
| `Prompts/` | Prompt templates, messages, and chatbot examples |
| `Runnables/` | Runnable sequences, passthrough, and branching |

`requirements.txt` contains the main LangChain integrations and supporting packages. The examples are independent, so each uses only the packages and external services it needs.

## Setup

Use a supported Python version and run these commands from the repository root:

```powershell
py -m venv myenv
.\myenv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The runnable examples in `Chains/` and `Runnables/` that use Ollama need its LangChain integration:

```powershell
python -m pip install langchain-ollama
```

They also require [Ollama](https://ollama.com/) to be installed and running locally, with the model tags used by the examples available. For example:

```powershell
ollama pull llama3.2:3b
```

Some scripts use hosted models and require provider credentials. Create a `.env` file in the repository root with only the credentials for the examples you intend to run:

```dotenv
GOOGLE_API_KEY=your-google-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
HUGGINGFACEHUB_API_TOKEN=your-hugging-face-token
```

Do not commit real credentials. The repository ignores `.env` files.

## Running examples

Activate the virtual environment, change to the repository root, and run a script by its path. For example:

```powershell
python ChatModels\chatmodel_gemini.py
python Chains\simple_chains.py
python EmbeddingModels\Doc_similarity.py
python outputs\opt_parsers\json_opt_parser.py
```

Choose a script in the folder that matches the concept you want to try. Many examples call a remote model or local Ollama model, so they may incur provider charges, require account access, or need a model downloaded locally.

