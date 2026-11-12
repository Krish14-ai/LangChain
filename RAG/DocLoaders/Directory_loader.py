from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path = r'RAG\Books',
    glob = '*.pdf',
    loader_cls=  PyPDFLoader
)

docs = loader.load() ## Eager Loading(Loads everything at once)

lazy_docs  = loader.lazy_load() ## Loads on demand and return a 'generator' of document object

print(docs[100].page_content)
print(docs[0].metadata)

for documnet in lazy_docs :
    print(documnet.metadata)