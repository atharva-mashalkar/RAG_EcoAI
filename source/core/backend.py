import os
import re
import fitz  # PyMuPDF
from typing import List
from pydantic import BaseModel, Field
import concurrent

# Import your LangChain modules (adjust import paths as needed)
# --- LangChain and Vector Store Imports ---
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings  # e.g., using BAAI/bge-large-en-v1.5
from langchain.vectorstores import Chroma

# --- Additional LangChain & Retrieval Imports ---
from langchain.chains.query_constructor.base import AttributeInfo, StructuredQueryOutputParser
from langchain.retrievers import SelfQueryRetriever
from langchain.retrievers.self_query.chroma import ChromaTranslator
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_groq import ChatGroq
from langchain.chains.query_constructor.base import get_query_constructor_prompt
from langchain.output_parsers.structured import StructuredOutputParser, ResponseSchema

# Memory imports
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

from config import PDF_FOLDER_PATH, PERSISTANT_DIRECTORY

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

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


vectorstore = Chroma(
    persist_directory=PERSISTANT_DIRECTORY,
    embedding_function=embeddings,
    collection_name="reports"
)



# ---------------------------
# Define the structured output model
# ---------------------------
class Citation(BaseModel):
    pdf_path: str = Field(..., description="The file path of the source.")
    page_number: int = Field(..., description="The page number from which the source was extracted.")

class CitedAnswer(BaseModel):
    answer: str = Field(
        ...,
        description="Detailed answer to the user question, which is based only on the given sources. Do not cite the sources in the answer.",
    )
    citations: List[Citation] = Field(
        ...,
        description="The citations from the sources (file path and page number) which justify the answer.",
    )


# ---------------------------
# PDF Processing and Metadata Extraction
# ---------------------------
def extract_metadata_from_filepath(filepath):
    country_match = re.search(r'OSE-([A-Za-z]+)', filepath)
    country = country_match.group(1) if country_match else "Unknown"
    year_match = re.search(r'\b(\d{4})\b', filepath)
    year = year_match.group(1) if year_match else "2024"
    return {"country": country, "year": year, "pdf_path": filepath}

def process_pdf_document(pdf_path, header_thresh=0.08, footer_thresh=0.05):
    """
    Opens a PDF and extracts text on a per-page basis.
    Only text blocks whose normalized vertical positions fall between header_thresh and (1 - footer_thresh)
    are kept. The returned dictionary maps page numbers (1-based) to text.
    """
    doc = fitz.open(pdf_path)
    pages_text = {}

    for page_number in range(len(doc)):
        page = doc[page_number]
        page_height = page.rect.height
        blocks = page.get_text("blocks")
        page_blocks = []

        for block in blocks:
            x0, y0, x1, y1, text, *_ = block
            if not text.strip():
                continue
            # Compute normalized top and bottom positions
            normalized_top = y0 / page_height
            normalized_bottom = y1 / page_height

            # Keep only body text (excluding headers/footers)
            if header_thresh <= normalized_top and normalized_bottom <= (1 - footer_thresh):
                page_blocks.append(text.strip())

        if page_blocks:
            # Page numbers are 1-based
            pages_text[page_number + 1] = "\n".join(page_blocks)
    return pages_text

# ---------------------------
# Load & Chunk Documents with Page-Level Metadata
# ---------------------------
def load_and_chunk_documents(folder_path, header_thresh=0.08, footer_thresh=0.):
    docs = []
    if not os.path.isdir(folder_path):
        print("❌ Provided path is not a directory.")
        return

    print("### Processing PDF documents in folder:")
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing {file_path}...")

            try:
                pages_text = process_pdf_document(file_path, header_thresh, footer_thresh)
                metadata = extract_metadata_from_filepath(file_path)
                # Create a Document for each page with corresponding metadata
                for page_number, text in pages_text.items():
                    doc_metadata = metadata.copy()
                    doc_metadata["page_number"] = page_number
                    docs.append(Document(page_content=text, metadata=doc_metadata))
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    print(f"### Loaded {len(docs)} documents with metadata.")

    # Chunk the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    print(f"### Created {len(chunks)} chunks.")

    # Remove duplicate chunks (by content)
    unique_chunks = list({chunk.page_content: chunk for chunk in chunks}.values())

    # Store chunks in your vector store (assumes vectorstore is already defined)
    vectorstore.add_documents(unique_chunks)
    print(f"✅ Successfully stored {len(unique_chunks)} chunks into Chroma.")

# ---------------------------
# Retrieval & Query Setup
# ---------------------------
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

def get_latest_year_for_country(country):
    docs = vectorstore.similarity_search("", filter={"country": country}, k=1000)
    if not docs:
        return None
    try:
        latest_year = max(int(doc.metadata.get("year", "0")) for doc in docs)
    except ValueError:
        latest_year = None
    return str(latest_year) if latest_year else None

def generate_country_specific_query(original_query, country):
    response_schemas = [
        ResponseSchema(name="query", description="Refined query focused on the given country.")
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    
    prompt_text = f"""
        Rewrite the following query to focus on {country}.
        Output only the refined query in JSON format as follows:
        {format_instructions}

        Original query: {original_query}
        """
    refined_output = llm.invoke(prompt_text).content
    parsed = output_parser.parse(refined_output)
    return parsed["query"]

def rewrite_query_with_history(current_input, memory):

    """
    Analyzes the current user input and conversation history to determine if:
    1. It's a new, independent question (return as-is)
    2. It's a follow-up that needs context from history (generate complete question)
    
    Args:
        current_input (str): The user's current input/question
        memory: The memory object containing conversation history
    
    Returns:
        str: Either the original input or a rewritten complete question
    """

    # Load chat history from memory
    chat_history_vars = memory.load_memory_variables({})
    chat_history = chat_history_vars.get("history", [])

    # If there's no history, return the original query
    if not chat_history:
        return current_input

    # Format chat history for analysis
    history_text = "\n".join([f"Human: {msg.content}" if isinstance(msg, HumanMessage) 
                              else f"AI: {msg.content}" for msg in chat_history])
    
    # Define the structured output schema for the analysis
    response_schemas = [
        ResponseSchema(name="query_type", description="Type of query: either 'NEW_QUESTION' (if it's an independent question not relying on previous context) or 'FOLLOW_UP' (if it requires context from previous conversation)"),
        ResponseSchema(name="rewritten_query", description="A complete, self-contained question that incorporates all necessary context from the conversation history. For NEW_QUESTION, just repeat the original input. For FOLLOW_UP, expand the query to include all context needed to understand it without seeing the conversation history.")
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    
    # Prompt with clear instructions, few-shot examples, and structured output format
    analysis_prompt = f"""
    Your task is to analyze a conversation history and current user input to:
    
    1. Determine if the current input is:
       - A NEW_QUESTION: Completely independent from the previous conversation
       - A FOLLOW_UP: Depends on context from the previous conversation (uses pronouns like "it", "they", "those", refers to previously mentioned topics, or asks for comparisons/more details about previously discussed subjects)
    
    2. If it's a FOLLOW_UP, rewrite the query to be a complete, standalone question by:
       - Replacing pronouns with their actual referents
       - Including all necessary context from previous turns
       - Making sure someone with no access to the conversation history would fully understand the question
       - Preserving the original intent of the question
    
    Output your analysis in JSON format according to these instructions:
    {format_instructions}
    
    Few-shot examples:
    
    Example 1:
    Chat History:
    Human: What are the key strengths of Qatar's entrepreneurial ecosystem?
    AI: Qatar's entrepreneurial ecosystem has several key strengths, including strong government support through initiatives like the Qatar Science & Technology Park, substantial funding availability from sovereign wealth funds, a strategic location as a gateway between East and West, and advanced infrastructure. The country has also developed specialized economic zones and provides tax incentives for startups and foreign investors.
    
    Current Input: What about its weaknesses?
    
    Output:

    {{
        "query_type": "FOLLOW_UP",
        "rewritten_query": "What are the weaknesses in Qatar's entrepreneurial ecosystem?"
    }}

    
    Example 2:
    Chat History:
    Human: What are the key strengths of Qatar's entrepreneurial ecosystem?
    AI: Qatar's entrepreneurial ecosystem has several key strengths, including strong government support through initiatives like the Qatar Science & Technology Park, substantial funding availability from sovereign wealth funds, a strategic location as a gateway between East and West, and advanced infrastructure. The country has also developed specialized economic zones and provides tax incentives for startups and foreign investors.
    Human: How does it compare to UAE?
    AI: Qatar and UAE's entrepreneurial ecosystems have similarities but also key differences. While both benefit from oil wealth and strategic locations, the UAE (particularly Dubai) has a more mature ecosystem with greater diversity in sectors, a larger expatriate population, more established free zones, and stronger international connections. Qatar focuses more on knowledge economy and education, with greater emphasis on research through institutions like Qatar Foundation.
    
    Current Input: Can you tell me about entrepreneurship in India?
    
    Output:

    {{
        "query_type": "NEW_QUESTION",
        "rewritten_query": "Can you tell me about entrepreneurship in India?"
    }}

    
    Example 3:
    Chat History:
    Human: What are the major challenges for entrepreneurs in Egypt?
    AI: Entrepreneurs in Egypt face numerous challenges including bureaucratic hurdles, limited access to financing, regulatory uncertainty, skills gaps in the workforce, infrastructure limitations, and macroeconomic instability. The country has been working to address these through reforms, but significant obstacles remain, particularly for small and medium enterprises.
    
    Current Input: How can these be addressed?
    
    Output:

    {{
        "query_type": "FOLLOW_UP",
        "rewritten_query": "How can the major challenges for entrepreneurs in Egypt (such as bureaucratic hurdles, limited financing, regulatory uncertainty, skills gaps, infrastructure limitations, and macroeconomic instability) be addressed?"
    }}


    Example 4:
    Chat History:
    Human: what are the strengths of egypts ecosystem
    AI: Egypt's entrepreneurial ecosystem has several strengths that contribute to its growth and development. Firstly, the ecosystem demonstrates moderate connectivity, with established actors well-connected, which facilitates access to resources and collaboration across sectors. Additionally, the country has a strong political leadership that has ensured stability and policy continuity, essential for fostering a conducive entrepreneurial environment. The ecosystem also shows promising collaboration, with a growing emphasis on education and skill development, and more programs supporting entrepreneurs and innovators. Furthermore, Egypt has positioned itself as a regional leader in digital entrepreneurship, particularly in fintech, e-commerce, and logistics, attracting both local and international investment. The country's young, tech-savvy population and its strategic location as a gateway to Africa and the Middle East also provide a competitive advantage. The COVID-19 crisis sparked rapid innovation, particularly in health tech and crisis management solutions, enhancing resilience. Overall, Egypt's entrepreneurial ecosystem has made significant strides in innovation, financing, and global cooperation, with a growing number of startups and entrepreneurs contributing to the country's economic growth and development. However, to sustain long-term growth, the ecosystem must address its challenges, such as regional disparities, resource integration, and policy alignment, and leverage its strengths to create a more resilient and inclusive ecosystem.

    Current Input: what about qatar
    Output:

    {{
        "query_type": "FOLLOW_UP",
        "rewritten_query": "What are the strengths of Qatar's entrepreneurial ecosystem?"

    }}
    

    
    Now analyze the following:
    
    Chat History:
    {history_text}
    
    Current Input: {current_input}
    """

    try:
        # Get structured analysis result
        analysis_result = llm.invoke(analysis_prompt).content
        parsed_result = output_parser.parse(analysis_result)
        if parsed_result["query_type"] == "NEW_QUESTION":
            # Return original input if it's a new question
            return current_input
        else:
            # Return the rewritten query that incorporates context
            return parsed_result["rewritten_query"]
            
    except Exception as e:
        # Fallback mechanism if parsing fails
        print(f"Warning: Error parsing structured output: {e}")
        # Simple fallback - just return the original query
        return current_input
    

def retrieve_documents(input_data):
    query = input_data["query"]
    
    country_keywords = ["Qatar", "EGYPT", "KSA", "UAE", "USA", "Morocco", "Germany"]
    mentioned_countries = [c for c in country_keywords if c.lower() in query.lower()]
    mentioned_years = re.findall(r"\b(19\d{2}|20\d{2}|2100)\b", query)
    
    retrieval_queries = []
    if mentioned_countries and not mentioned_years:
        for country in mentioned_countries:
            latest_year = get_latest_year_for_country(country)
            if latest_year:
                filter_criteria = f'and(eq("country", "{country}"), eq("year", "{latest_year}"))'
                # filter_criteria = f'and(eq("country", ["{country}"]), eq("year", ["{latest_year}"]))'
            else:
                filter_criteria = f'eq("country", "{country}")'
                # filter_criteria = f'eq("country", ["{country}"])'
            refined_query = generate_country_specific_query(query, country)
            retrieval_queries.append((refined_query, filter_criteria))
    elif mentioned_years:
        for year in mentioned_years:
            retrieval_queries.append((query, f'eq("year", ["{year}"])'))
    if not retrieval_queries:
        retrieval_queries.append((query, None))
    
    retrieved_docs = []
    for rq, filter_criteria in retrieval_queries:
        # print("filter criteria:", filter_criteria)
        params = {"query": rq}
        if filter_criteria:
            params["filter"] = filter_criteria
        retrieved_docs.extend(retriever.invoke(params))
    
    return {"context": retrieved_docs, "question": query}


# ---------------------------
# Format Documents with Citation IDs
# ---------------------------
def format_docs(docs):
    # Assign a citation (source) ID to each retrieved document.
    formatted_texts = []
    for idx, doc in enumerate(docs, start=1):
        metadata = doc.metadata
        # Each source is formatted with its assigned citation ID,
        # and includes the PDF path and page number.
        formatted_texts.append(
            f"Source: {idx}\nContent: {doc.page_content}\nCountry: {metadata['country']}\nYear: {metadata['year']}\nCitation: (PDF: {metadata['pdf_path']}, Page {metadata['page_number']})"
        )
    return "\n\n".join(formatted_texts)


  
def get_conversation_chain():
    # Create a prompt template WITHOUT the chat history
    # prompt_template = ChatPromptTemplate.from_template(
    #     "Use only the following context to answer the question.\n\n"
    #     "Provide clear, detailed and well-structured answers.\n"
    #     "Answer using more than 500 words.\n\n"
    #     "Context:\n{context}\n\nQuestion: {question}"
    # )

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", 
         """You are an expert analyst. Use ONLY the provided context to answer the question.
         Context contains sources with citations like (PDF: path.pdf, Page X). 
         
         Your response MUST be in this format:
         {{
             "answer": "Detailed answer using context...",
             "citations": [
                 {{"pdf_path": "file1.pdf", "page_number": 1}},
                 {{"pdf_path": "file2.pdf", "page_number": 5}}
             ]
         }}
         
         Rules:
         1. Answer must be 500+ words
         2. NEVER invent citations - use only those from context
         3. ALWAYS include EXACTLY matching PDF paths from context
         """),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])
    
    # The chain uses our formatting function to build context, then passes it to the final prompt,
    # and finally uses our structured LLM to output a CitedAnswer.
    structured_llm = llm.with_structured_output(CitedAnswer)
    
    def format_context(input_dict):
        # Format the documents
        context_docs = input_dict["context"]
        formatted_context = format_docs(context_docs)
        
        # Return components needed for the prompt
        return {
            "context": formatted_context,
            "question": input_dict["question"]
        }
    
    def debug_print_prompt(input_dict):
        """Helper function to print the final prompt before passing to LLM."""
        final_prompt = prompt_template.format(**input_dict)
        print("\n===== FINAL PROMPT =====\n")
        print(final_prompt)
        print("Length of prompt:", len(final_prompt))
        print("\n========================\n")
        return input_dict  # Pass input unchanged to maintain chain flow

    # The complete chain with debug print
    chain = (
        RunnablePassthrough.assign(formatted_input=format_context)
        | (lambda x: {"context": x["formatted_input"]["context"], 
                      "question": x["formatted_input"]["question"]})
        # | debug_print_prompt  # Print the final prompt
        | prompt_template
        | structured_llm
    )
    
    return chain



def process_query(query):
    # Step 1: Rewrite the query if it's a follow-up question
    rewritten_query = rewrite_query_with_history(query, memory)
    print("Rewritten query:\n", rewritten_query)
    
    # Step 2: Retrieve relevant documents using the rewritten query
    retrieved_docs = retrieve_documents({"query": rewritten_query})
    # print("Docs:\n", retrieved_docs)
    # Step 3: Get the conversation chain with memory
    conversation_chain = get_conversation_chain()
    
    # Step 4: Process through the chain
    result = conversation_chain.invoke(retrieved_docs)
    
    # Step 5: Save the original interaction to memory
    memory.save_context(
        {"input": query},  # Save original query to maintain natural conversation flow
        {"output": result.answer}
    )
    
    return result.answer, result.citations

# ---------------------------
# Main Execution Functions
# ---------------------------
def initialize_system(pdf_folder_path):
    """Initialize the system by loading documents if needed"""
    # Uncomment the next line to process and chunk PDFs before retrieval.
    # load_and_chunk_documents(pdf_folder_path, header_thresh=0.08, footer_thresh=0.05)
    print("System initialized and ready to process queries!")

def chat_loop():
    """Interactive chat loop for the RAG system"""
    print("Welcome to the Entrepreneurial Ecosystem RAG System!")
    print("Type 'exit' or 'quit' to end the conversation.")
    
    while True:
        user_input = "what are the strengths of the entrepreneurial ecosystem in Qatar?"  #input("\nYour question: ")
        
        if user_input.lower() in ['exit', 'quit']:
            print("Thank you for using the system. Goodbye!")
            break
            
        print("\nProcessing your question...")
        try:
            result = process_query(user_input)
            
            print("\n----- ANSWER -----")
            print(result.answer)
            print("\n----- CITATIONS -----")
            for citation in result.citations:
                print(f"- {citation.pdf_path}, Page {citation.page_number}")
            print("\n")
            
        except Exception as e:
            print(f"Error processing your question: {e}")
            import traceback
            traceback.print_exc()


initialize_system(PDF_FOLDER_PATH)



# Initialize Conversation Summary Buffer Memory
# This will summarize older parts of the conversation to save tokens


# ---------------------------
# Run the Pipeline
# ---------------------------
# if __name__ == "__main__":
#     pdf_folder_path = "documents"  # Update as needed
#     initialize_system(pdf_folder_path)
    
#     # For a single query test
#     # query = "what are the strengths of the entrepreneurial ecosystem in Qatar?"
#     # result = process_query(query)
#     # print("Answer:")
#     # print(result.answer)
#     # print("\nCitations:")
#     # for citation in result.citations:
#     #     print(f"- {citation.pdf_path}, Page {citation.page_number}")
        
#     # For interactive chat
#     chat_loop()
