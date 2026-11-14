from langchain_community.document_loaders import CSVLoader

path = r'RAG\Docs\black_holes.csv'

loader = CSVLoader(file_path=path)

docs = loader.load()

print(docs[0].page_content)