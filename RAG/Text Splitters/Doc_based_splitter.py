import warnings
warnings.filterwarnings("ignore", message=".*langchain-community.*sunset.*")

from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import DirectoryLoader

path = r'RAG\Text Splitters'

loader = DirectoryLoader(
    path = path,
    glob = '*.py'
)

files = loader.load()

splitter = RecursiveCharacterTextSplitter.from_language(
    language= Language.PYTHON,
    chunk_size = 50,
    chunk_overlap = 15
)

result = splitter.split_documents(files)

print(result)