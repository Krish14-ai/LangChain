from langchain_ollama import ChatOllama 
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOllama(
    model="llama3.2:3b",
    temperature=0.7
)

prompt = PromptTemplate(
    template = "write me a short note on the {topic}",
    input_variables=['topic']
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'topic' : 'cricket'})

chain.get_graph().print_ascii()

print(result)