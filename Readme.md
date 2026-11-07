# My Journey with LangChain

I’m using this repository to learn how LangChain can help me build applications with large language models. Each folder contains small experiments as I explore a new concept, try different models, and connect the pieces into useful workflows.

My learning starts with the basics: sending prompts to chat models and shaping their responses. From there, I’m exploring how to manage conversation messages, combine steps into chains, and use LangChain’s Runnable interface to compose and branch workflows. I’m also experimenting with embeddings to represent and compare text, and with structured outputs and parsers to get responses in formats my code can work with.

## Concepts I’m exploring

- **Chat models:** Working with models from providers such as Gemini, Anthropic, Hugging Face, and Ollama.
- **Prompts and conversation messages:** Creating reusable prompts and including chat history in model requests.
- **Chains and Runnables:** Connecting model calls and other operations into sequential, parallel, or conditional workflows.
- **Embeddings and similarity:** Converting text into vectors and comparing them to find related content.
- **Structured outputs:** Using schemas, Pydantic models, and output parsers to organize model responses.

## Inside the repository

- `ChatModels/` — chat model experiments
- `Prompts/` and `messeges/` — prompts, messages, and conversation history
- `Chains/` — multi-step and conditional workflows
- `Runnables/` — composing and branching workflow steps
- `EmbeddingModels/` — text embeddings and similarity experiments
- `outputs/` — structured responses and output parsers

This is a work in progress. The examples capture what I’m learning along the way, and I’ll keep building on them as I get more comfortable designing LLM workflows with LangChain.