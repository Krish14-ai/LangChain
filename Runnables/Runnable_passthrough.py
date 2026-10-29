from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSequence, RunnableParallel

model= ChatOllama(
    model = 'llama3.2:3b',
    temperature=2
)

prompt_1 = PromptTemplate(
    template = 'Give me a joke on {topic}',
    input_variables=['topic']
)

prompt_2 = PromptTemplate(
    template = 'Explain this joke :- \n {joke}',
    input_variables=['joke']
)

parser = StrOutputParser()

pass_chain = RunnableSequence(prompt_1, model, parser)

joke_chain = RunnableSequence( prompt_2, model, parser)

chain = RunnablePassthrough.assign(
    joke = pass_chain
) | RunnableParallel({
    'topic':lambda x : x['topic'],
    'joke' : lambda x : x['joke'],
    'explanation' : joke_chain
    }) 

result = chain.invoke({'topic' : 'politics'})
print(result)


