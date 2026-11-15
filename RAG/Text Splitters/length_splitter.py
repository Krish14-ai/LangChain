from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

path = r'RAG\Docs'

loader = DirectoryLoader(
    path = path,
    
)