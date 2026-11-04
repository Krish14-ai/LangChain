from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOllama(
    model = 'llama3.2:3b',
    temperature = 1.25
)

prompt = PromptTemplate(
    template='Summarize the text :{text}',
    input_variables=['text']
)

parser = StrOutputParser()
loader = PyPDFLoader(r'RAG\Docs\black_holes.pdf')

docs = loader.load()

# print(docs,'\n')

# print(docs[0].page_content,'\n')

# print(docs[0].metadata)

chain = prompt | model | parser

result = chain.invoke({'text' : docs[0].page_content})

print(result)