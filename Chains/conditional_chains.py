from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableBranch
from pydantic import BaseModel, Field
from typing import Literal

model = ChatOllama(
    model="llama3.2:3b",
)

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="The sentiment of the feedback"
    )

pydantic_parser = PydanticOutputParser(
    pydantic_object=Feedback
)

prompt_1 = PromptTemplate(
    template=(
        "Classify the sentiment of the following feedback. "
        "Return only the required structured output.\n"
        "Feedback: {feedback}\n"
        "{format_instruction}"
    ),
    input_variables=["feedback"],
    partial_variables={
        "format_instruction": pydantic_parser.get_format_instructions()
    }
)

classifier_chain = prompt_1 | model | pydantic_parser

result = classifier_chain.invoke({
    "feedback": "this is a dumb ass phone"
})

print(result)
print(result.sentiment)