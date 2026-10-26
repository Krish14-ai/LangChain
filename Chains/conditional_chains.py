from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
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
parser = StrOutputParser()

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

## prompt for positive feedback
prompt_pos = PromptTemplate(
    template = "write an appropriate response to this positive feedback \n {feedback}",
    input_variables= ['feedback']
)

## prompt for negative feedback
prompt_neg = PromptTemplate(
    template = "write an appropriate response to this negative feedback \n {feedback}",
    input_variables= ['feedback']
)


branch_chain = RunnableBranch(
    (lambda x : x.sentiment == 'positive', prompt_pos | model | parser), # (condition, chain)
    (lambda x : x.sentiment == 'negative', prompt_neg | model | parser),
    RunnableLambda(lambda x : "couldn't find any sentiment")
)

final_chain = classifier_chain | branch_chain

final_result =final_chain.invoke({'feedback' : 'this is a good phone'})

print(final_result)