import os
from dotenv import load_dotenv
from config import PERSISTANT_DIRECTORY
from langchain.vectorstores import Chroma
from langchain.memory import ConversationSummaryBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings  # e.g., using BAAI/bge-large-en-v1.5
from langchain_groq import ChatGroq
from langchain.chains.query_constructor.base import AttributeInfo, StructuredQueryOutputParser
from langchain.chains.query_constructor.base import get_query_constructor_prompt
from langchain.retrievers import SelfQueryRetriever
from langchain.retrievers.self_query.chroma import ChromaTranslator

load_dotenv('config/.env')

llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("GROQ_API_KEY")
)

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=2000,  # Adjust based on your needs
    return_messages=True,
)

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

vectorstore = Chroma(
    persist_directory=PERSISTANT_DIRECTORY,
    embedding_function=embeddings,
    collection_name="reports"
)

document_content_description = "Yearly reports on Structural and Configurational Analysis of Entrepreneurial Ecosystems for different countries."
metadata_field_info = [
    AttributeInfo(name="year", description="The year of the report.", type="string"),
    AttributeInfo(name="country", description="The country of the report.", type="string"),
]

prompt = get_query_constructor_prompt(document_content_description, metadata_field_info)
output_parser = StructuredQueryOutputParser.from_components()
query_constructor_runnable = prompt | llm | output_parser

retriever = SelfQueryRetriever(
    query_constructor=query_constructor_runnable,
    vectorstore=vectorstore,
    structured_query_translator=ChromaTranslator(),
    search_kwargs={'k': 2},
)

