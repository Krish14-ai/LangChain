from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

path = r'RAG\Books\Agentic-Artificial-Intelligence.pdf'

loader = PyPDFLoader(file_path= path)

docs = loader.load()

text = docs[101].page_content


splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
    separator= ''
)


# print(splitter.split_text(text))

result = splitter.split_documents(docs)

print(result[0])

