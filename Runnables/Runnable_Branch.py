from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableBranch, RunnableSequence,RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

model  = ChatOllama(
    model = 'llama3.2:3b',
    temperature=2
)

prompt_1 = PromptTemplate(
    template = 'explain this topic :- {topic}',
    input_variables= ['topic']
)

prompt_2 = PromptTemplate(
    template = 'Summarize this :- {report}',
    input_variables= ['report']
)

parser = StrOutputParser()

report_gen_chain = RunnableSequence(
    prompt_1, model, parser
)

branch_cain = RunnableBranch(
    (lambda x : len(x.split()) > 500, RunnableSequence(prompt_2, model, parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain, branch_cain)

result = final_chain.invoke({'topic' : 'rusia'})

print(result)