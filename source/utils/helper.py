import re
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.output_parsers.structured import StructuredOutputParser
from source.utils.response_schema import *
from source.utils.setup import *
from source.utils.prompts import *

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

    response_schemas = country_specific_response_schema
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    prompt = country_specific_query_prompt.format(format_instructions=format_instructions, original_query=original_query, country=country)
    refined_output = llm.invoke(prompt).content
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
    response_schemas = query_rewriter_schema
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    
    # Prompt with clear instructions, few-shot examples, and structured output format
    prompt = query_rewriter_prompt.format(format_instructions=format_instructions, history_text=history_text, current_input=current_input)

    try:
        # Get structured analysis result
        analysis_result = llm.invoke(prompt).content
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
            else:
                filter_criteria = f'eq("country", "{country}")'
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

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", final_answer_prompt),
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