
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

model1 = ChatOllama(
    model="llama3.2:3b",
    temperature=0.7
)

model2 = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=1
)

prompt_1 = PromptTemplate(
    template="Generate short and simple notes from the following text:\n{topic}",
    input_variables=["topic"]
)

prompt_2 = PromptTemplate(
    template="Generate 5 short questions and answers from the following text:\n{text}",
    input_variables=["text"]
)

prompt_3 = PromptTemplate(
    template="""
Merge the following notes and quiz into a single document.

NOTES:
{notes}

QUIZ:
{quiz}
""",
    input_variables=["notes", "quiz"]
)

parser = StrOutputParser()

path = r"C:\Users\Krish\Downloads\LangChain\Chains\notes.txt"

with open(path, "r", encoding="utf-8") as f:
    text = f.read()

parallel_chain = RunnableParallel({
    "notes": prompt_1 | model1 | parser,
    "quiz": prompt_2 | model2 | parser
})

merge_chain = prompt_3 | model1 | parser

chain = parallel_chain | merge_chain

# result = chain.invoke({
#     "topic": text,
#     "text": text
# })

# print(result)

chain.get_graph().print_ascii()