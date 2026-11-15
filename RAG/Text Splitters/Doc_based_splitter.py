from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import DirectoryLoader

path = r'RAG\Text Splitters'

loader = DirectoryLoader(
    path = path,
    glob = '*.py'
)