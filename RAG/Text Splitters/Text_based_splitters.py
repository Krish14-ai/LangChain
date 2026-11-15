from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

path  = r'RAG\Docs\black_holes.pdf'

loader = PyPDFLoader(
    file_path=path
)

doc = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 10,
    chunk_overlap = 0,
    separators= [" "]
)

result = splitter.split_documents(doc)

print(result[0])