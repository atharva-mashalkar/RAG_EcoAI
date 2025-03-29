from config import PDF_FOLDER_PATH
from source.utils import *


# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


# ---------------------------
# Retrieval & Query Setup
# ---------------------------


def process_query(query):
    try:

        # Route the Query based on its a general greeting or needs to be answered using the databsse
        route , response = route_query(query)
        if not route:
            return response, []

        # Step 1: Rewrite the query if it's a follow-up question
        rewritten_query = rewrite_query_with_history(query, memory)
        print("Rewritten query:\n", rewritten_query)
        
        # Step 2: Retrieve relevant documents using the rewritten query
        retrieved_docs = retrieve_documents({"query": rewritten_query})

        # Step 3: If no docuemnts are retrieved ⁠⁠suggest a meeting with https://calendly.com/gregory-gueneau
        if not retrieved_docs:
            return "No relevant documents found. Please consider scheduling a meeting with https://calendly.com/gregory-gueneau for further assistance." , []

        # Step 4: Get the conversation chain with memory
        conversation_chain = get_conversation_chain()
        
        # Step 5: Process through the chain
    
        result = conversation_chain.invoke(retrieved_docs)

        # Step 6: Save the original interaction to memory
        memory.save_context(
            {"input": query},  # Save original query to maintain natural conversation flow
            {"output": result.answer}
        )
        
        return result.answer, result.citations
    
    except Exception as e:
        return "Internal Server Error. Please consider scheduling a meeting with https://calendly.com/gregory-gueneau for further assistance." , [f"Error:{e}"]


# ---------------------------
# Main Execution Functions
# ---------------------------
def initialize_system(pdf_folder_path):
    """Initialize the system by loading documents if needed"""
    # Uncomment the next line to process and chunk PDFs before retrieval.
    # load_and_chunk_documents(pdf_folder_path, vectorstore, header_thresh=0.08, footer_thresh=0.05)
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
