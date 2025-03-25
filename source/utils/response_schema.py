from typing import List
from pydantic import BaseModel, Field
from langchain.output_parsers.structured import StructuredOutputParser, ResponseSchema

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


country_specific_response_schema = [
        ResponseSchema(name="query", description="Refined query focused on the given country.")
    ]

query_rewriter_schema = [
        ResponseSchema(name="query_type", description="Type of query: either 'NEW_QUESTION' (if it's an independent question not relying on previous context) or 'FOLLOW_UP' (if it requires context from previous conversation)"),
        ResponseSchema(name="rewritten_query", description="A complete, self-contained question that incorporates all necessary context from the conversation history. For NEW_QUESTION, just repeat the original input. For FOLLOW_UP, expand the query to include all context needed to understand it without seeing the conversation history.")
    ]