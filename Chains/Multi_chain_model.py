from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature = 0
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

