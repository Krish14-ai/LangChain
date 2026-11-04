## Text Loader is a simple and commonly used document loader in Langchain that reads plain text files and converts them into Langchain Document Objects

from langchain_community.document_loaders import TextLoader

path = r'RAG\Docs\black_holes.txt'

loader = TextLoader(file_path=path, encoding= 'utf-8')

docs = loader.load()

print(docs)

print('\n', type(docs))

