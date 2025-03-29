import os
import re
import fitz  # PyMuPDF
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from source.utils.setup import *


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