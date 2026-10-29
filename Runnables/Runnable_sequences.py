from langchain_ollama import ChatOllama 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel

prompt_1 = PromptTemplate(
    template = "write me a poem on {topic}",
    input_variables= ['topic']
)

prompt_2 = PromptTemplate(
    template='explain this porm :- \n{poem}',
    input_variables= ['poem']
)

model = ChatOllama(
    model = 'llama3.2:3b',
    temperature=2
)

parser = StrOutputParser()

chain = RunnableSequence(prompt_1, model, parser, prompt_2, model, parser)
result = chain.invoke({'topic' : 'Moon'})

print(result)

