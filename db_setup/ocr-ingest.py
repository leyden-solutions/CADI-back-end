import os
import glob
import tiktoken
from typing import List, Dict, Any
from dataclasses import dataclass
import pytesseract
from pdf2image import convert_from_path
from datetime import datetime
import json
from pathlib import Path
import logging
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

from openai import OpenAI
from supabase import create_client
import numpy as np

# Load environment variables
load_dotenv()

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Set up logging
def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"pdf_processing_{timestamp}.log")
    
    # Configure logging
    logger = logging.getLogger("PDFProcessor")
    logger.setLevel(logging.DEBUG)
    
    # File handler with detailed formatting
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler with simpler formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

@dataclass
class PDFChunk:
    content: str
    page_number: int
    chunk_index: int
    tokens: int
    metadata: Dict[str, Any]

class PDFProcessor:
    def __init__(self, openai_api_key: str, supabase_url: str, supabase_key: str):
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.supabase = create_client(supabase_url, supabase_key)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.chunk_size = 1500
        self.chunk_overlap = 200
        self.logger = logging.getLogger("PDFProcessor")
        # Tesseract configuration options
        self.tesseract_config = '--oem 3 --psm 6'  # Page segmentation mode: 6 - Assume a single uniform block of text

    def extract_text_from_pdf(self, file_path: str) -> List[str]:
        """Extract text from PDF file using Tesseract OCR, returning a list of pages."""
        self.logger.info(f"Starting text extraction from {file_path} using Tesseract OCR")
        try:
            # Convert PDF to images
            self.logger.info("Converting PDF to images")
            pdf_images = convert_from_path(
                file_path, 
                dpi=300,  # Higher DPI for better OCR quality
                thread_count=4  # Use multiple threads for faster conversion
            )
            
            total_pages = len(pdf_images)
            self.logger.info(f"PDF converted to {total_pages} images")
            
            pages = []
            for page_num, image in enumerate(tqdm(pdf_images, desc="OCR processing", unit="page")):
                try:
                    # Use pytesseract to extract text from the image
                    page_text = pytesseract.image_to_string(image, config=self.tesseract_config)
                    # Clean up the text (remove extra whitespace, etc.)
                    page_text = " ".join(page_text.split())
                    pages.append(page_text)
                    self.logger.debug(f"Successfully extracted text from page {page_num + 1}")
                except Exception as e:
                    self.logger.error(f"Error extracting text from page {page_num + 1}: {str(e)}")
                    pages.append("")  # Add empty string for failed pages
            
            self.logger.info(f"Completed text extraction. Extracted {len(pages)} pages")
            return pages
        except Exception as e:
            self.logger.error(f"Failed to process PDF file {file_path}: {str(e)}")
            raise

    def create_chunks(self, pages: List[str]) -> List[PDFChunk]:
        """Create overlapping chunks from PDF pages."""
        self.logger.info("Starting chunk creation")
        chunks = []
        chunk_index = 0
        
        for page_num, page in enumerate(tqdm(pages, desc="Creating chunks", unit="page"), 1):
            try:
                tokens = self.tokenizer.encode(page)
                self.logger.debug(f"Page {page_num}: {len(tokens)} tokens")
                
                if len(tokens) <= self.chunk_size:
                    chunks.append(PDFChunk(
                        content=page,
                        page_number=page_num,
                        chunk_index=chunk_index,
                        tokens=len(tokens),
                        metadata={"source": f"page_{page_num}"}
                    ))
                    chunk_index += 1
                    continue
                
                # Split larger pages into chunks
                current_chunk = []
                current_tokens = 0
                
                for token in tokens:
                    current_chunk.append(token)
                    current_tokens += 1
                    
                    if current_tokens >= self.chunk_size:
                        chunk_text = self.tokenizer.decode(current_chunk)
                        chunks.append(PDFChunk(
                            content=chunk_text,
                            page_number=page_num,
                            chunk_index=chunk_index,
                            tokens=current_tokens,
                            metadata={"source": f"page_{page_num}"}
                        ))
                        chunk_index += 1
                        
                        # Keep overlap tokens for next chunk
                        current_chunk = tokens[max(0, len(current_chunk) - self.chunk_overlap):]
                        current_tokens = len(current_chunk)
                
                # Don't forget the last chunk of the page
                if current_tokens > 0:
                    chunk_text = self.tokenizer.decode(current_chunk)
                    chunks.append(PDFChunk(
                        content=chunk_text,
                        page_number=page_num,
                        chunk_index=chunk_index,
                        tokens=current_tokens,
                        metadata={"source": f"page_{page_num}"}
                    ))
                    chunk_index += 1
                    
            except Exception as e:
                self.logger.error(f"Error processing page {page_num}: {str(e)}")
        
        self.logger.info(f"Completed chunk creation. Created {len(chunks)} chunks")
        return chunks

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts using OpenAI's API."""
        self.logger.info(f"Starting embedding generation for {len(texts)} chunks")
        embeddings = []
        
        # Process in batches of 100
        batch_size = 100
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings", unit="batch"):
            batch = texts[i:i + batch_size]
            try:
                # Handle empty strings and very short texts
                processed_batch = [text if text.strip() else "Empty content" for text in batch]
                
                # The API expects each text to be a string
                response = self.openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=[str(text) for text in processed_batch]  # Ensure all inputs are strings
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
                self.logger.debug(f"Successfully generated embeddings for batch {i//batch_size + 1}")
            except Exception as e:
                self.logger.error(f"Error generating embeddings for batch {i//batch_size + 1}: {str(e)}")
                raise
        
        self.logger.info("Completed embedding generation")
        return embeddings

    def process_pdf(self, file_path: str, category: str) -> Dict[str, Any]:
        """Process a single PDF file and store it in Supabase."""
        start_time = time.time()
        self.logger.info(f"Starting processing of {file_path}")
        
        try:
            # Extract basic document info
            pages = self.extract_text_from_pdf(file_path)
            file_name = Path(file_path).name
            
            # Insert document record
            doc_data = {
                "file_name": file_name,
                "file_path": str(Path(file_path).absolute()),
                "category": category,
                "total_pages": len(pages),
                "metadata": {
                    "processed_date": datetime.utcnow().isoformat(),
                    "file_size": os.path.getsize(file_path),
                    "extraction_method": "tesseract_ocr"
                }
            }
            
            self.logger.info("Inserting document record")
            doc_response = self.supabase.table("documents").insert(doc_data).execute()
            document_id = doc_response.data[0]["id"]
            
            # Process chunks
            chunks = self.create_chunks(pages)
            
            # Get embeddings for all chunks
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = self.get_embeddings(chunk_texts)
            
            # Prepare chunk records
            self.logger.info("Preparing chunk records")
            chunk_records = []
            for chunk, embedding in zip(chunks, embeddings):
                chunk_record = {
                    "document_id": document_id,
                    "chunk_index": chunk.chunk_index,
                    "page_number": chunk.page_number,
                    "content": chunk.content,
                    "tokens": chunk.tokens,
                    "embedding": embedding,
                    "metadata": chunk.metadata
                }
                chunk_records.append(chunk_record)
            
            # Insert chunks in batches
            batch_size = 100
            total_batches = (len(chunk_records) + batch_size - 1) // batch_size
            
            self.logger.info(f"Inserting {len(chunk_records)} chunks in {total_batches} batches")
            for i in tqdm(range(0, len(chunk_records), batch_size), desc="Inserting chunks", unit="batch"):
                batch = chunk_records[i:i + batch_size]
                try:
                    self.supabase.table("chunks").insert(batch).execute()
                    self.logger.debug(f"Successfully inserted batch {i//batch_size + 1}/{total_batches}")
                except Exception as e:
                    self.logger.error(f"Error inserting batch {i//batch_size + 1}: {str(e)}")
                    raise
            
            processing_time = time.time() - start_time
            self.logger.info(f"Completed processing {file_name} in {processing_time:.2f} seconds")
            
            return {
                "success": True,
                "document_id": document_id,
                "total_chunks": len(chunks),
                "total_tokens": sum(c.tokens for c in chunks),
                "processing_time": processing_time
            }
            
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path
            }

def process_pdf_directory(
    pdf_dir: str,
    category: str,
    openai_api_key: str,
    supabase_url: str,
    supabase_key: str
) -> List[Dict[str, Any]]:
    """Process all PDFs in a directory."""
    # Set up logging
    logger = setup_logging()
    logger.info(f"Starting batch processing of PDFs in {pdf_dir}")
    
    processor = PDFProcessor(openai_api_key, supabase_url, supabase_key)
    results = []
    
    # Get all PDF files in directory
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs", unit="file"):
        logger.info(f"Starting processing of {pdf_file}")
        result = processor.process_pdf(pdf_file, category)
        results.append(result)
        
        if result["success"]:
            logger.info(f"Successfully processed {pdf_file}")
            logger.info(f"Chunks: {result['total_chunks']}, Tokens: {result['total_tokens']}")
        else:
            logger.error(f"Failed to process {pdf_file}: {result['error']}")
    
    return results

if __name__ == "__main__":
    # Load environment variables
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    
    if not all([OPENAI_API_KEY, SUPABASE_URL, SUPABASE_KEY]):
        print("Error: Missing required environment variables")
        exit(1)
    
    # Example usage
    pdf_directory = "sop_pdfs"
    category = "military_docs"
    
    results = process_pdf_directory(
        pdf_directory,
        category,
        OPENAI_API_KEY,
        SUPABASE_URL,
        SUPABASE_KEY
    )
    
    # Print summary
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    print("\nProcessing Summary:")
    print(f"Successfully processed: {successful} files")
    print(f"Failed to process: {failed} files")
    
    if successful > 0:
        total_chunks = sum(r["total_chunks"] for r in results if r["success"])
        total_tokens = sum(r["total_tokens"] for r in results if r["success"])
        total_time = sum(r["processing_time"] for r in results if r["success"])
        print(f"\nTotal chunks created: {total_chunks}")
        print(f"Total tokens processed: {total_tokens}")
        print(f"Total processing time: {total_time:.2f} seconds")