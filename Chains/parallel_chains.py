from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

model1 = ChatOllama(
    model="llama3.2:3b",
    temperature=0.7
)

model2 =ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=1
)
prompt_1 = PromptTemplate(
    template = "Generate short and simple notes from the following text \n {topic}",
    input_variables=['topic']
)

prompt_2 = PromptTemplate(
    template  = 'Generate 5 short Questions and answers from the following {text}',
    input_variables= ['text']
)

