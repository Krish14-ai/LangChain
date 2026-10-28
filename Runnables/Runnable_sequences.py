from langchain_ollama import ChatOllama 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

prompt = PromptTemplate(
    template = "write me a poem on {topic}",
    input_variables= ['topic']
)

model = ChatOllama(
    model = 'llama3.2:3b',
    temperature=2
)
