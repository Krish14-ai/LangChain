## Text Loader is a simple and commonly used document loader in Langchain that reads plain text files and converts them into Langchain Document Objects

from langchain_community.document_loaders import TextLoader
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

path = r'RAG\Docs\black_holes.txt'

model = ChatOllama(
    model="llama3.2:3b",
    temperature= 1.5
)

parser = StrOutputParser()



loader = TextLoader(file_path=path, encoding= 'utf-8')

docs = loader.load()

prompt=PromptTemplate(template= "write a summary of this text : {text}",
                        input_variables=['text'])

print('\n', type(docs))

# print(docs[0])
# print(type(docs[0]),'\n')
# print(docs[0].metadata)


## Chain 
chain = prompt | model | parser

result = chain.invoke({'text' : docs[0].page_content})

print(result)