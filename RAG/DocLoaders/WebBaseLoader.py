from langchain_community.document_loaders import  WebBaseLoader

url = "https://www.amazon.in/realme-Obsidian-6300mAh-Segments-Biggest/dp/B0H1BMBX9R/ref=sr_1_3?crid=EYMG42IC8EV6&dib=eyJ2IjoiMSJ9.0ocGqvVUCk8UGb_YlhOsWyi6YZHa3D3eqDJbnsbFyvf9stbMwVJTBZOgVaJ8fOBqRlsEs58b7tHZdvMOYvlNhSMlRYExVbA5lT1YRrS5gOWmXGFoJ7SVoL4k5GAbHtLFkFa7icpwJBRtreRSMFfMYVz7ZtCOmAiHfnlQPzwPW9MqP_8Rb7QVUnFamT7jd95ty_boSaTEfqndymXOV9vn7oXn9tssQDYsMQr1nEAbEcY.v7xDqAVWX__0yjycrNaBiwbO3OH948miuaOBRdtiU28&dib_tag=se&keywords=phones%2Bunder%2B15%2Bk&qid=1794476217&sprefix=phones%2Caps%2C372&sr=8-3&th=1"

loader = WebBaseLoader(url)

docs = loader.load()

print(len(docs))

print(docs[0].page_content)

