from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatOllama(
    model="llama3.2:3b",
    temperature=0.7
)


parser = StrOutputParser()

template_1 = PromptTemplate(
    template = "Explain {topic} in detail",
    input_variables=['topic']
)

template_2 = PromptTemplate(
    template = "Extract 5 most important things from this :- \n {details}",
    input_variables=['details']
)

chain_1 = template_1 | model | parser
chain_2 = template_2 | model | parser

chain = {"details": chain_1} | chain_2

result = chain.invoke({"topic": "LangGraph"})

print(result)

chain.get_graph().print_ascii()