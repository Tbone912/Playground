import chromadb
import ollama
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from pypdf import PdfReader

documents = []

reader = PdfReader('example.pdf')
page = reader.pages[0]

for page_number in range(5,20):
    page = reader.pages[page_number]
    documents.append(page.extract_text())



client = chromadb.Client()
collection = client.create_collection(name="docs")

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma(collection_name="docs", embedding_function=embeddings)



for i, doc in enumerate(documents):
    vectorstore.add_texts([doc], metadatas=[{"source": f"doc{i}"}], ids=[f"id{i}"])

llm = Ollama(model="llama3.1:latest")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

query = "Does this dryer steam with a hot water line?"
response = qa_chain.run(query)

print("/n/n" + response + "/n/n")
