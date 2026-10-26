from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal

model = ChatOllama(
    model="llama3.2:3b",
    temperature=1
)

parser = StrOutputParser()

prompt_1 = PromptTemplate(
    template = "classsify the sentiment of the following feedback as only Positive or Negative also dont add any other text : \n {feedback}",
    input_variables= ['feedback']
)

classifier_chain = prompt_1 | model | parser 

print(classifier_chain.invoke({'feedback' : 'this is a Good phone'}))