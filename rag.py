import ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from pypdf import PdfReader

# Load PDF documents
documents = []
reader = PdfReader('example.pdf')
for page_number in range(5,6):
    page = reader.pages[page_number]
    documents.append(page.extract_text())

# Create a Chroma vector store
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(
    collection_name="docs",
    embedding_function=embeddings
)

# Add documents to the vector store
vectorstore.add_texts(
    texts=documents,
    metadatas=[{"source": f"doc{i}"} for i in range(len(documents))],
    ids=[f"id{i}" for i in range(len(documents))]
)

# Initialize the LLM
llm = Ollama(model="llama3.1:latest")

# Create a RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Run a query
query = "Which way do I turn the bulb to change the light and what watt?"
response = qa_chain.run(query)

print("\n\n" + response + "\n\n")
