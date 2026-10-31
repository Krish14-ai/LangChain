# Learning LangChain Prompts

This folder is part of my LangChain learning journey. It explores how to give a model instructions, fill prompts with values, work with chat messages, and turn a prompt into part of a runnable model workflow.

## What is in this folder?

- **`template.py`** creates a string `PromptTemplate` for summarizing a research paper. The template has three inputs: `paper_input`, `style_input`, and `length_input`. It saves the prompt as `template_1.json`.
- **`template_1.json`** is the serialized prompt. It preserves the template text and its input-variable metadata so it can be loaded again.
- **`Prompts_ui.py`** loads that saved prompt, collects the paper, style, and length from Streamlit controls, and pipes the filled prompt into a Gemini model with `template | model`.
- **`messages.py`** builds a conversation from `SystemMessage` and `HumanMessage`, sends it to a model, and adds the response as an `AIMessage`. This demonstrates role-based messages rather than a reusable prompt template.
- **`chatbot.py`** uses Streamlit to keep and display chat history, then sends the conversation to a Gemini chat model. Its message history gives the model context between turns.

## Prompt-template patterns

LangChain prompts generally fall into two useful shapes: **text prompts** for a single string input to a model, and **chat prompts** made of role-based messages for chat models. The examples below show the common patterns and how they relate to this folder.

### 1. `PromptTemplate` — a reusable text prompt

Use this when the model should receive one formatted string. Placeholders such as `{topic}` are filled when the prompt is invoked.

```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    "Explain {topic} in a {style} way."
)
formatted = prompt.invoke({"topic": "embeddings", "style": "beginner-friendly"})
```

`template.py` uses this pattern, with three named inputs and a longer instruction for a research-paper summary.

### 2. Saved prompts — reusing a serialized template

A prompt can be saved and loaded so the template can be reused separately from the Python code that calls the model:

```python
from langchain_core.prompts import load_prompt

prompt = load_prompt("template_1.json")
```

`Prompts_ui.py` follows this pattern. The JSON file stores the prompt and metadata; it is still a text `PromptTemplate`, not a separate kind of prompt.

### 3. `ChatPromptTemplate` — a reusable role-based chat prompt

Use this to define a chat conversation as a sequence of system, human, and assistant messages. Variables can be placed inside any message.

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful {subject} tutor."),
    ("human", "{question}"),
])
messages = prompt.invoke({
    "subject": "science",
    "question": "What is an embedding?",
})
```

`messages.py` demonstrates role-based messages directly, while `chatbot.py` builds messages from live conversation history. Neither currently uses `ChatPromptTemplate`; this example shows the reusable-template version of that idea.

### 4. `MessagesPlaceholder` — inserting conversation history

When a chat prompt needs a variable number of earlier messages, use a placeholder instead of treating the history as one text string.

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])
```

The [`messeges/message_place_holder.py`](../messeges/message_place_holder.py) example elsewhere in this repository explores this pattern. In this folder, `chatbot.py` instead passes its Streamlit message history directly to the chat model.

### 5. `FewShotPromptTemplate` — examples inside a text prompt

Few-shot prompting provides example inputs and outputs so a model can follow a demonstrated pattern.

```python
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

example_prompt = PromptTemplate.from_template("Input: {input}\nOutput: {output}")
prompt = FewShotPromptTemplate(
    examples=[
        {"input": "happy", "output": "positive"},
        {"input": "broken", "output": "negative"},
    ],
    example_prompt=example_prompt,
    prefix="Classify the sentiment:",
    suffix="Input: {text}\nOutput:",
    input_variables=["text"],
)
```

This pattern is not currently used by the examples in this folder.

### 6. `FewShotChatMessagePromptTemplate` — examples as chat turns

For chat models, few-shot examples can be represented as human/assistant message pairs and inserted into a larger chat prompt.

```python
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)

examples = [
    {"question": "What is 2 + 2?", "answer": "4"},
]
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{question}"),
    ("ai", "{answer}"),
])
few_shot = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
)
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer questions clearly."),
    few_shot,
    ("human", "{question}"),
])
```

This pattern is not currently used by the examples in this folder.

### 7. Partial prompts — pre-filling some variables

When part of a prompt stays the same across calls, pre-fill those variables and provide only the remaining values at invocation time.

```python
prompt = PromptTemplate.from_template(
    "Explain {topic} for a {audience} audience."
).partial(audience="beginner")

result = prompt.invoke({"topic": "LangChain"})
```

Partial variables also appear in the serialized prompt format, although `template_1.json` currently has none.

## How the research-prompt example fits together

The research-summary exercise follows this flow:

1. `template.py` defines instructions and the three variables.
2. The prompt is saved to `template_1.json`.
3. `Prompts_ui.py` loads the saved prompt and gathers values from the user.
4. `template | model` formats the prompt and sends it to Gemini.
5. The generated summary is displayed in the Streamlit app.

The style and length choices are values passed into the prompt; they are not different prompt-template types. The instruction also asks the model not to guess when information is unavailable.

## Running the examples

From the repository root, run the Streamlit examples with:

```powershell
streamlit run Prompts\chatbot.py
streamlit run Prompts\Prompts_ui.py
```

`Prompts_ui.py` loads `template_1.json` by a relative path, so launch Streamlit from the repository root. It also depends on the Gemini credentials and packages configured for the project.

To regenerate the saved template, run:

```powershell
python Prompts\template.py
```

This writes `template_1.json` to the current working directory. Run it from the repository root if you want to update the file in `Prompts/`.
