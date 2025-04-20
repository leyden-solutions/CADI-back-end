# Standard library imports
import base64
import gc
import io
import json
import logging
import os
import psutil
import re
import tempfile
import uuid
from datetime import datetime
import time
from typing import Dict, Any, List

# Third party imports
import anthropic
import jwt
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pdf2image import convert_from_bytes, convert_from_path
from PIL import Image, ImageDraw
from pydantic import BaseModel
import pytesseract
from supabase import create_client, Client, ClientOptions
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PyPDF2 import PdfReader

from prompt import guidelines_prompt, sections_prompt

# Load environment variables from .env file
load_dotenv()

# Get the JWT secret key from environment variables
JWT_SECRET = os.getenv("JWT_SECRET")

if not JWT_SECRET:
    raise ValueError("JWT_SECRET not found in environment variables")

prompt_string = """
You are tasked with generating a list of redactions for a given list of words based on a set of rules. Your goal is to identify sections of text that need to be redacted according to the provided rules and output a list of non-overlapping redaction intervals. The user will provide the list of words, the rules are provided in this system prompt.

Here is the list of rules to apply:
<rules>
{RULES}
</rules>

Follow these steps to complete the task:

1. Read through the list of words carefully. Consider that these words may form complete sentences or parts of sentences.

2. Review each rule in the provided list. Pay attention to the rule number, grouping, level, and content description.

3. For each rule, scan the list of words to identify any sections that match or closely relate to the rule's content description.

4. When you find a match, determine the start and end indices for the section that should be redacted. The start index should be the index of the first word to be redacted, and the end index should be the index of the last word to be redacted.

5. After determining the start and end indices for redactions, also create a list of indices to skip in those redactions to avoid redacting words that do not need to be redacted, or words that won't leak sensitive information.

6. Keep in mind that you are processing one page at a time. There may be text preceding the beginning or following the end of the given list of words. When deciding on redactions near the beginning or end of the list, consider what the likely context would be and redact accordingly.

7. Ensure that your redaction intervals do not overlap. If you find overlapping intervals, merge them into a single, larger interval that covers all the words that need to be redacted.

8. Create a list of redaction objects. Each object should have the following properties:
   - start: the index of the first word to be redacted (integer, minimum 0)
   - end: the index of the last word to be redacted (integer, maximum len(words)-1)
   - rule: the exact rule identifier from the rules list that triggered this redaction (use the "rule" field from the matching rule)
   - level: the exact level specified in the matching rule (use the "level" field from the matching rule)
   - reasoning: a brief explanation of why this section matches the rule (string)
   - confidence: a float between 0 and 1 indicating the confidence of the match. if it would be < 0.2, do not add it to the list (float)
   - skip: a list of indices between start and end to skip in this redaction (list of integers)

9. Sort the list of redaction objects by the start index in ascending order.

Output your final list of redactions in the following format:
<redactions>
[{{start:index(int),end:index(int),skip:(list[int]),rule:rule_id_from_rules,level:level_from_rules,reasoning:"brief explanation", confidence:probability(float)}},
{{start:index(int),end:index(int),skip:(list[int]),rule:rule_id_from_rules,level:level_from_rules,reasoning:"brief explanation", confidence:probability(float)}},
{{start:index(int),end:index(int),skip:(list[int]),rule:rule_id_from_rules,level:level_from_rules,reasoning:"brief explanation", confidence:probability(float)}},
...]
</redactions>

Here's an example of what your output might look like:
<example>
<redactions>
[{{start:0,end:5,skip:[1,2],rule:"{EXAMPLE_RULE_1}",level:"{EXAMPLE_LEVEL_1}",reasoning:"This section contains personal identifying information matching rule 1", confidence:0.9}},
{{start:10,end:15,skip:[12,14],rule:"{EXAMPLE_RULE_2}",level:"{EXAMPLE_LEVEL_2}",reasoning:"This content describes sensitive company operations prohibited by rule 2", confidence:0.82}},
{{start:20,end:25,skip:[21,22,23,24],rule:"{EXAMPLE_RULE_3}",level:"{EXAMPLE_LEVEL_3}",reasoning:"These words match keywords associated with proprietary technology in rule 3", confidence:0.41}}]
</redactions>
</example>

Remember:
- NO OVERLAPPING INTERVALS in your final output
- Use ONLY the exact rule identifiers and levels as they appear in the rules list
- The start index should be at minimum 0
- The end index should be at maximum len(words)-1
"""

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'ocr_server_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure pytesseract path
pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_PATH")

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Security scheme for JWT
security = HTTPBearer()

# Model for JWT response
class JWTData(BaseModel):
    payload: Dict[str, Any]

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL or SUPABASE_KEY not found in environment variables")

def decode_jwt(token: str) -> dict:
    """
    Decode and validate JWT token and check if user is allowed in credits table
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"], audience="authenticated")

        # Create Supabase client
        supabase: Client = create_client(
            SUPABASE_URL,
            SUPABASE_KEY,
            ClientOptions(headers={"Authorization": f"Bearer {token}"})
        )

        # Check credits table for user's allowed status
        result = supabase.table("credits").select("allowed").eq("user_id", payload["sub"]).execute()

        if not result.data or not result.data[0].get("allowed"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User is not allowed to access this resource"
            )

        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

def get_memory_usage() -> float:
    """Get current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def process_pdf(pdf_bytes: bytes, doc_id: str | None = None) -> Dict:
    """Process PDF and return structured results with bounding boxes"""
    logger.info("Starting PDF processing with bounding boxes")
    logger.info(f"Initial memory usage: {get_memory_usage():.2f} MB")

    results = []
    page_count = 0

    try:
        # Create a temporary file to store the PDF bytes
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
            temp_pdf.write(pdf_bytes)
            temp_pdf_path = temp_pdf.name

        try:
            while True:
                page_count += 1
                logger.info(f"Processing page {page_count}")
                logger.info(f"Memory usage before page {page_count}: {get_memory_usage():.2f} MB")

                # Convert only the current page
                images = convert_from_path(
                    temp_pdf_path,
                    first_page=page_count,
                    last_page=page_count,
                    dpi=300
                )

                # If no images returned, we've reached the end of the PDF
                if not images:
                    page_count -= 1  # Adjust for the last increment
                    break

                img = images[0]  # Get the single page image
                width, height = img.size

                # Convert PIL image to bytes for OCR
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()

                logger.debug(f"Performing OCR with bounding boxes on page {page_count}")

                # Get detailed data including bounding boxes
                ocr_data = pytesseract.image_to_data(
                    Image.open(io.BytesIO(img_byte_arr)),
                    output_type=pytesseract.Output.DICT,
                    config=r'--psm 3'
                )

                # Process OCR data
                n_boxes = len(ocr_data['text'])
                ct = 0
                for j in range(n_boxes):
                    if not ocr_data['text'][j].strip():
                        continue

                    conf = float(ocr_data['conf'][j])
                    if conf < 0:
                        continue

                    x1 = ocr_data['left'][j] / width
                    y1 = ocr_data['top'][j] / height
                    x2 = (ocr_data['left'][j] + ocr_data['width'][j]) / width
                    y2 = (ocr_data['top'][j] + ocr_data['height'][j]) / height

                    result = {
                        'text': ocr_data['text'][j],
                        'num': ct,
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2,
                        'page': page_count
                    }
                    if doc_id:
                        result["doc_id"] = doc_id
                    results.append(result)
                    ct += 1

                # Clear memory for the current page
                del images
                del img
                del img_byte_arr
                gc.collect()

                logger.info(f"Memory usage after page {page_count}: {get_memory_usage():.2f} MB")

        finally:
            os.unlink(temp_pdf_path)

    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise

    logger.info(f"Final memory usage: {get_memory_usage():.2f} MB")
    return {"page_count": page_count, "words": results}

def convert_to_json(input_string):
    # Add quotes around property names
    step1 = re.sub(r'([{,])\s*([a-zA-Z0-9_]+):', r'\1"\2":', input_string)

    # Convert JavaScript object to valid JSON
    json_obj = eval(f"[{step1.strip('[]')}]")

    return json_obj

def process_redactions(doc_id: str, page: int, user_id: str, supabase: Client):
    """
    Process and upload redactions for a specific page of a document.

    Args:
        doc_id: The document ID

        page: The page number (1-indexed)
        content: The page content to analyze for redactions
        supabase: Supabase client instance
    """
    try:
        # Initialize Anthropic client
        client = anthropic.Anthropic()

        #get the guideline id
        guideline_response = supabase.table("docs").select("guideline_id").eq("doc_id", doc_id).execute()
        guideline_id = guideline_response.data[0]["guideline_id"]

        # Get classification rules from database
        rules_response = supabase.table("declass_rules").select("rule, grouping, level, content").neq("level", "U").eq("guideline_id", guideline_id).execute()
        rules = rules_response.data

        words_response = supabase.table("words") \
            .select("num,text") \
            .eq("doc_id", doc_id) \
            .eq("page", page) \
            .execute()

        words = words_response.data

        # Get example values for the template
        example_rule_1 = rules[0]["rule"]  # "3.3.1-28"
        example_level_1 = rules[0]["level"]  # "S"
        example_rule_2 = rules[1]["rule"]  # "3.3.1-29"
        example_level_2 = rules[1]["level"]  # "S"
        example_rule_3 = rules[2]["rule"]  # "3.3.1-30"
        example_level_3 = rules[2]["level"]  # "S"

        final_prompt = prompt_string.format(
            RULES=rules,
            EXAMPLE_RULE_1=example_rule_1,
            EXAMPLE_LEVEL_1=example_level_1,
            EXAMPLE_RULE_2=example_rule_2,
            EXAMPLE_LEVEL_2=example_level_2,
            EXAMPLE_RULE_3=example_rule_3,
            EXAMPLE_LEVEL_3=example_level_3
        )

        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=2048,
            system=[
            {
                "type": "text",
                "text": final_prompt,
                "cache_control": {"type": "ephemeral"}
            }
            ],
            messages=[{"role": "user", "content": json.dumps(words)}],
        )


        # Extract redactions from response
        redactions_match = re.search(r'<redactions>\s*(.*?)\s*</redactions>', response.content[0].text, re.DOTALL)
        redactions_only = redactions_match.group(1)
        redactions = convert_to_json(redactions_only)
        for r in redactions:
            skip = set(r["skip"])
            r["indices"] = [i for i in range(r["start"], r["end"]+1) if i not in skip]
            r["start"] = r["indices"][0]
            r["end"] = r["indices"][-1]
        redactions = [{"doc_id": doc_id, "page": page, "status": "Pending", "user_id": user_id, "reasoning": r["reasoning"], "start": r["start"], "end": r["end"], "rule": r["rule"], "confidence": r["confidence"], "indices": r["indices"]} for r in redactions]
        supabase.table("redactions").insert(redactions).execute()

    except Exception as e:
        logger.error(f"Error processing redactions for doc {doc_id} page {page}: {str(e)}")
        raise e

async def upload_doc(
    doc_id: str,
    file_content: bytes,
    filename: str,
    guideline_id: str,
    supabase: Client,
    uid: str
):
    """
    Upload a PDF document to Supabase storage and create a document record.
    Links the document to a guideline document.
    """

    try:

        # Get paths
        file_path = f"{uid}/{doc_id}"
        filepath_pdf = f"{file_path}.pdf"

        images = convert_from_bytes(file_content)
        page_count = len(images)

        # Convert each image to base64
        images_bytes = []
        for img in images:
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()
            images_bytes.append(img_bytes)

        try:
            # Upload file to storage first
            supabase.storage.from_("docs").upload(
                path=filepath_pdf,
                file=file_content,
                file_options={"content-type": "application/pdf"}
            )

            # Upload the images to storage
            for i, img_bytes in enumerate(images_bytes):
                image_path = f"{file_path}/{i+1}.png"
                supabase.storage.from_("doc-images").upload(
                    path=image_path,
                    file=img_bytes,
                    file_options={"content-type": "image/png"}
                )

            # If storage upload succeeds, create database record
            doc_data = {
                "doc_id": doc_id,
                "guideline_id": guideline_id,
                "filepath": file_path,
                "filename": filename,  # Store original filename
                "status": "Analyzing",
                "page_count":page_count  # Add page count here
            }

            supabase.table("docs").insert(doc_data).execute()
            supabase.table("docs_users").insert({"doc_id": doc_id, "user_id": uid, "type": "owner"}).execute()

            return page_count

        except Exception as e:
            # If database insert fails, try to clean up the storage
            try:
                supabase.storage.from_("docs").remove(file_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup storage file: {cleanup_error}")
                pass  # Best effort cleanup
            raise e

    except Exception as e:
        logger.error(f"Error in upload_doc: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

async def process_document(
    doc_id: str,
    uid: str,
    supabase: Client
):
    """
    Process a document that has been uploaded to Supabase storage.
    Updates the page count, word count, and status in the docs table.
    """
    try:
        # Get the filepath from the docs table
        result = supabase.table("docs").select("filepath,filename").eq("doc_id", doc_id).execute()
        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )

        filepath = result.data[0]["filepath"]
        filepath_pdf = f"{filepath}.pdf"

        # Get the file from storage
        try:
            file_data = supabase.storage.from_("docs").download(filepath_pdf)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found in storage"
            )

        # Process PDF and extract text with bounding boxes
        logger.info(f"Processing PDF for document {doc_id}")
        result = process_pdf(file_data, doc_id)
        page_count = result["page_count"]
        words_data = result["words"]

        # Insert words in batches to avoid request size limits
        batch_size = 1000
        for i in range(0, len(words_data), batch_size):
            batch = words_data[i:i + batch_size]
            supabase.table("words").insert(batch).execute()

        logger.info(f"Successfully processed and stored {len(words_data)} words")

        # Process redactions for each page
        for i in range(1, page_count + 1):
            process_redactions(doc_id, i, uid, supabase=supabase)
            supabase.table("docs").update({"pages_processed": i}).eq("doc_id", doc_id).execute()
            yield f"data: {json.dumps({'doc_id': doc_id, 'status': 'Update', 'pages_processed': i})}\n\n"

        # Update document record with word count and status
        supabase.table("docs").update({"word_count": len(words_data), "status": "Ready"}).eq("doc_id", doc_id).execute()

    except Exception as e:
        logger.error(f"Error in process_document: {str(e)}")
        # Update document status to error
        try:
            supabase.table("docs").update({"status": "Error"}).eq("doc_id", doc_id).execute()
        except Exception as status_error:
            logger.warning(f"Failed to update document status: {status_error}")
            pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

async def event_generator_documents(uid: str, supabase: Client, file_content: bytes, filename: str, guideline_id: str):
    """Generate status events for document processing. Basically a wrapper for the upload_doc and process_document functions."""
    try:
        # Upload document and get document ID and filename

        doc_id = str(uuid.uuid4())


        yield f"data: {json.dumps({'doc_id': doc_id, 'status': 'Uploading', 'filename': filename})}\n\n"


        # Upload the document
        try:
            page_count = await upload_doc(doc_id, file_content, filename, guideline_id, supabase, uid)
        except Exception as e:
            logger.error(f"Error uploading document: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )

        yield f"data: {json.dumps({'doc_id': doc_id, 'filename': filename, 'status': 'Analyzing', 'page_count': page_count})}\n\n"


        # Process the document
        try:
            async for event in process_document(doc_id, uid, supabase):
                yield event
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )

        # Send a completion event for processing completion
        yield f"data: {json.dumps({'doc_id': doc_id, 'status': 'Ready', 'filename': filename})}\n\n"

    except Exception as e:
        logger.error(f"Error in event_generator: {str(e)}")
        # Send an error event
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/upload-process-doc")
async def upload_process_doc(
    file: UploadFile = File(...),
    guideline_id: str = Form(...),
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    """
    Upload a PDF document and process it.
    Returns a streaming response with updates.
    """
    token = credentials.credentials
    payload = decode_jwt(token)
    uid = payload.get("sub")

    supabase: Client = create_client(
        SUPABASE_URL,
        SUPABASE_KEY,
        ClientOptions(headers={"Authorization": f"Bearer {token}"})
    )

    file_content = await file.read()
    filename = file.filename

    return StreamingResponse(
        event_generator_documents(uid, supabase, file_content, filename, guideline_id),
        media_type="text/event-stream"
    )


@app.get("/get-pdf/{doc_id}")
async def get_pdf(
    doc_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Get a PDF document from Supabase storage using the document ID
    """
    token = credentials.credentials
    decode_jwt(token)

    try:
        supabase: Client = create_client(
            SUPABASE_URL,
            SUPABASE_KEY,
            ClientOptions(headers={"Authorization": f"Bearer {token}"})
        )

        result = supabase.table("docs").select("filepath,page_count").eq("doc_id", doc_id).execute()

        if not result.data or len(result.data) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )

        filepath = result.data[0]["filepath"]
        page_count = result.data[0]["page_count"]
        filepath_pdf = f"{filepath}.pdf"

        try:
            file_data = supabase.storage.from_("docs").download(filepath_pdf)
            file_b64 = base64.b64encode(file_data).decode("utf-8")
            return {
                "file": file_b64,
                "page_count": page_count
            }

        except Exception:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found in storage"
            )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/page-data/{doc_id}")
async def get_page_data(
    doc_id: str,
    page: int,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Get words and redactions for a specific page of a document
    """
    token = credentials.credentials
    decode_jwt(token)

    try:
        supabase: Client = create_client(
            SUPABASE_URL,
            SUPABASE_KEY,
            ClientOptions(headers={"Authorization": f"Bearer {token}"})
        )

        words_result = (
            supabase.table("words")
            .select("text,x1,y1,x2,y2")
            .eq("doc_id", doc_id)
            .eq("page", page)
            .order("num")
            .execute()
        )

        rules_result = supabase.table("declass_rules").select("rule,level,content").execute()
        rule_level_dct = {r["rule"]: r["level"] for r in rules_result.data}
        rule_content_dct = {r["rule"]: r["content"] for r in rules_result.data}

        if not rules_result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No rules found"
            )

        redactions_result = (
            supabase.table("redactions")
            .select("start,end,rule,reasoning,status,user_id,redact_id,confidence,indices,user_id,last_updated")
            .eq("doc_id", doc_id)
            .eq("page", page)
            .execute()
        )

        if redactions_result.data:
            redactions = redactions_result.data
            # Gather all unique user_ids from redactions
            user_ids = list(set(r["user_id"] for r in redactions if r.get("user_id")))
            email_map = {}
            if user_ids:
                # Query credits table for all user_ids
                credits_result = supabase.table("credits").select("user_id,email").in_("user_id", user_ids).execute()
                if credits_result.data:
                    email_map = {c["user_id"]: c["email"] for c in credits_result.data}
            for r in redactions:
                r["level"] = rule_level_dct.get(r["rule"], None)
                r["content"] = rule_content_dct.get(r["rule"], None)
                r["email"] = email_map.get(r.get("user_id"), None)
            redactions.sort(key=lambda x: x["start"])

        words = words_result.data if words_result.data else []

        return {
            "words": words,
            "redactions": redactions
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )



def get_groupings_from_pdf(file_content: bytes):

    client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            types.Part.from_bytes(
                data=file_content,
                mime_type='application/pdf',
            ),
            sections_prompt
        ]
    )
    text = response.text
    text = text.replace("```json", "").replace("```", "")
    json_dict = json.loads(text)
    if "sections" in json_dict:
        return json_dict['sections']
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No sections found"
        )


def process_pdf_with_gemini(file_content: bytes, sections: str):
    """
    Process a PDF file using Gemini's model and return the generated content.
    """

    client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

    prompt = guidelines_prompt.replace("{sections}", sections)

    # Generate response from Gemini
    response = client.models.generate_content(
        model="gemini-2.5-pro-exp-03-25",
        contents=[
            types.Part.from_bytes(
                data=file_content,
                mime_type='application/pdf',
            ),
            prompt
        ]
    )
    text = response.text
    text = text.replace("```json", "").replace("```", "")

    return json.loads(text)


async def extract_rules_from_pdf(file_content: bytes, guideline_id: str, groupings: list):
    """
    Get rules from a PDF file using Gemini's model
    """
    batch_size = 5
    rules_data = []

    for i in range(0, len(groupings), batch_size):
        logger.info(f"Processing batch {i//batch_size + 1}")
        batch_groupings = groupings[i:i+batch_size]
        rules_dict = process_pdf_with_gemini(file_content, f"[{', '.join(batch_groupings)}]")
        # rules_dict = json.loads(example_response)
        for grouping, rule_data in rules_dict.items():
            for idx, item in enumerate(rule_data.get("items", []), 1):
                rules_data.append({
                    "guideline_id": guideline_id,
                    "rule": f"{grouping}-{idx}",
                    "grouping": grouping,
                    "content": item.get("content", ""),
                    "level": item.get("level", ""),
                    "duration": item.get("duration"),
                    "reason": item.get("reason", ""),
                    "remarks": item.get("remarks", "")
                })
            logger.info(f"Groupings processed: {min(i+batch_size, len(groupings))}")
            yield f"data: {json.dumps({'status': 'Update', 'guideline_id': guideline_id, 'groupings_processed': min(i+batch_size, len(groupings))})}\n\n"
    yield f"data: {json.dumps({'status': 'Complete', 'rules': rules_data})}\n\n"


async def event_generator_guidelines(uid: str, supabase: Client, file_content: bytes, filename: str):
    """Generate status events for guideline processing. Basically a wrapper for the upload_guideline function."""

    guideline_id = str(uuid.uuid4())

    yield f"data: {json.dumps({'guideline_id': guideline_id, 'status': 'Uploading', 'filename': filename})}\n\n"

    try:
        filepath = f"{uid}/{guideline_id}"
        filepath_pdf = f"{filepath}.pdf"
        try:
            supabase.storage.from_("guidelines").upload(
                path=filepath_pdf,
                file=file_content,
                file_options={"content-type": "application/pdf"}
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to upload file to storage: {str(e)}"
            )
        print("File uploaded successfully")

        # Get page count from PDF using PyPDF2 (much faster than other methods)
        pdf = PdfReader(io.BytesIO(file_content))
        page_count = len(pdf.pages)
        groupings = get_groupings_from_pdf(file_content)
        logger.info(f"Groupings: {groupings}")

        yield f"data: {json.dumps({'guideline_id': guideline_id, 'status': 'Analyzing', 'page_count': page_count, 'grouping_count': len(groupings)})}\n\n"

        # Create guideline record
        try:
            guideline_data = {
                "guideline_id": guideline_id,
                "filename": filename,
                "filepath": filepath,
                "status": "Ready",
                "page_count": page_count,
                "grouping_count": len(groupings)
            }
            supabase.table("guidelines").insert(guideline_data).execute()
        except Exception as e:
            # Cleanup the uploaded file if record creation fails
            try:
                supabase.storage.from_("guidelines").remove([filepath_pdf])
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup guideline file: {cleanup_error}")
                pass
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create guideline record: {str(e)}"
            )

        # Process and collect rules
        try:
            rules_data = []
            async for event in extract_rules_from_pdf(file_content, guideline_id, groupings):
                event_data = json.loads(event.replace("data: ", ""))
                if event_data.get("status") == "Complete":
                    rules_data = event_data.get("rules", [])
                else:
                    yield event

            if rules_data:
                supabase.table("declass_rules").insert(rules_data).execute()
            else:
                raise Exception("No rules were generated")

        except Exception as e:
            # Cleanup the uploaded file and guideline record if rule creation fails
            try:
                supabase.storage.from_("guidelines").remove([filepath_pdf])
                supabase.table("guidelines").delete().eq("guideline_id", guideline_id).execute()
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup guideline and record: {cleanup_error}")
                pass
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create rules: {str(e)}"
            )

        yield f"data: {json.dumps({'guideline_id': guideline_id, 'status': 'Ready', 'filename': filename})}\n\n"

    except Exception as e:
        logger.error(f"Error in event_generator: {str(e)}")
        # Send an error event
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )



@app.post("/upload-guideline")
async def upload_guideline(
    file: UploadFile = File(...),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Upload a PDF guideline document to Supabase storage and create related rules
    """
    token = credentials.credentials
    payload = decode_jwt(token)
    uid = payload.get("sub")

    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY, ClientOptions(headers={"Authorization": f"Bearer {token}"}))

    # Read the uploaded file
    file_content = await file.read()

    # Get filename
    filename = file.filename

    return StreamingResponse(
        event_generator_guidelines(uid, supabase, file_content, filename),
        media_type="text/event-stream"
    )



@app.get("/")
async def root():
    """
    Root endpoint with basic instructions
    """
    return {
        "message": "JWT Test API",
        "instructions": "Send a request to /test with a valid JWT token in the Authorization header"
    }

@app.get("/get-stats")
async def get_stats(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Get document and page statistics for the authenticated user
    Returns:
    - total_documents: Total number of documents
    - total_pages: Sum of page_count across all documents
    - declassifiable_pages: Total pages minus redacted pages
    - declassifiable_percentage: Percentage of words that are not redacted
    """
    token = credentials.credentials
    decode_jwt(token)

    try:
        supabase: Client = create_client(
            SUPABASE_URL,
            SUPABASE_KEY,
            ClientOptions(headers={"Authorization": f"Bearer {token}"})
        )

        docs_result = supabase.table("docs").select("*").neq("status", "Pending").execute()
        total_documents = len(docs_result.data)

        if total_documents == 0:
            return {
                "total_documents": 0,
                "total_pages": 0,
                "declassifiable_pages": 0,
                "declassifiable_percentage": 0
            }

        total_pages = sum(doc["page_count"] for doc in docs_result.data)
        total_words = sum(doc["word_count"] for doc in docs_result.data)

        redactions_result = (
            supabase.table("redactions")
            .select("doc_id, page")
            .execute()
        )

        redacted_pages = len(set((r['doc_id'], r['page']) for r in redactions_result.data))
        declassifiable_pages = total_pages - redacted_pages

        all_redactions = supabase.table("redactions").select("indices").execute()
        redacted_words = sum(len(r['indices']) for r in all_redactions.data)

        declassifiable_percentage = ((total_words - redacted_words) / total_words * 100) if total_words > 0 else 0
        return {
            "total_documents": total_documents,
            "total_pages": total_pages,
            "declassifiable_pages": declassifiable_pages,
            "declassifiable_percentage": round(declassifiable_percentage, 2)
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/get-chart-stats")
async def get_chart_stats(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Get declassification statistics for charts:
    - Overall declassifiable percentage
    - Declassifiable percentages for 5 most recent documents
    """
    token = credentials.credentials
    decode_jwt(token)

    try:
        supabase: Client = create_client(
            SUPABASE_URL,
            SUPABASE_KEY,
            ClientOptions(headers={"Authorization": f"Bearer {token}"})
        )

        docs_result = (
            supabase.table("docs")
            .select("*")
            .neq("status", "Pending")
            .order("created_at", desc=True)
            .execute()
        )
        total_words = sum(doc["word_count"] for doc in docs_result.data)

        all_redactions = supabase.table("redactions").select("doc_id,indices").execute()
        redacted_words_per_doc = {}
        for r in all_redactions.data:
            if r['doc_id'] not in redacted_words_per_doc:
                redacted_words_per_doc[r['doc_id']] = 0
            redacted_words_per_doc[r['doc_id']] += len(r['indices'])
        redacted_words = sum(redacted_words_per_doc.values())

        declassifiable_percentage = 0 if total_words == 0 else round(
            ((total_words - redacted_words) / total_words) * 100,
            2
        )

        documents = []
        for doc in docs_result.data[:5]:
            doc_total = doc["word_count"]
            doc_redacted = sum(
                len(r['indices'])
                for r in all_redactions.data
                if r['doc_id'] == doc['doc_id']
            )

            doc_declassifiable = 0 if doc_total == 0 else round(
                ((doc_total - doc_redacted) / doc_total) * 100,
                2
            )
            doc_classified = 100 - doc_declassifiable

            documents.append({
                "name": doc["filename"],
                "declassifiable": doc_declassifiable,
                "classified": doc_classified
            })

        return {
            "declassifiable_percentage": declassifiable_percentage,
            "documents": documents
        }

    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/get-documents")
async def get_documents(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Get all documents for the authenticated user
    """
    token = credentials.credentials
    decode_jwt(token)

    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY, ClientOptions(headers={"Authorization": f"Bearer {token}"}))

        # Get all documents
        result = supabase.table("docs").select("doc_id,filename,created_at,page_count,status").execute()

        return result.data

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/get-guidelines")
async def get_guidelines(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    decode_jwt(token)

    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY, ClientOptions(headers={"Authorization": f"Bearer {token}"}))

        # Get all guidelines with more fields
        result = supabase.table("guidelines").select("guideline_id,filename,created_at,status,page_count").execute()

        return result.data

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/get-guideline-pdf/{guideline_id}")
async def get_guideline_pdf(
    guideline_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Get a guideline PDF document from Supabase storage using the guideline ID
    """
    token = credentials.credentials
    decode_jwt(token)

    try:
        supabase: Client = create_client(
            SUPABASE_URL,
            SUPABASE_KEY,
            ClientOptions(headers={"Authorization": f"Bearer {token}"})
        )

        # Get the filepath from the guidelines table
        result = supabase.table("guidelines").select("filepath,filename").eq("guideline_id", guideline_id).execute()

        if not result.data or len(result.data) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Guideline not found"
            )

        filepath = result.data[0]["filepath"]
        filename = result.data[0]["filename"]
        filepath_pdf = f"{filepath}.pdf"

        try:
            # Get the file from storage
            file_data = supabase.storage.from_("guidelines").download(filepath_pdf)

            # Return the file as a downloadable response
            return StreamingResponse(
                io.BytesIO(file_data),
                media_type="application/pdf",
                headers={
                    "Content-Disposition": f'attachment; filename="{filename}"'
                }
            )

        except Exception:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found in storage"
            )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/get-rules/{doc_id}")
async def get_rules(
    doc_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Get rules associated with a document via its guideline ID
    """
    token = credentials.credentials
    decode_jwt(token)

    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY, ClientOptions(headers={"Authorization": f"Bearer {token}"}))

        # Get the guideline_id for this document
        doc_result = supabase.table("docs").select("guideline_id").eq("doc_id", doc_id).execute()

        if not doc_result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )

        guideline_id = doc_result.data[0]["guideline_id"]

        # Get all rules for this guideline
        rules_result = supabase.table("declass_rules").select("*").eq("guideline_id", guideline_id).execute()

        return rules_result.data

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/get-pdf-image/{doc_id}")
async def get_pdf_image(
    doc_id: str,
    page: int,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Convert a PDF document to a list of base64-encoded images
    """
    token = credentials.credentials
    decode_jwt(token)


    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY, ClientOptions(headers={"Authorization": f"Bearer {token}"}))

        # Get the PDF from storage
        result = supabase.table("docs").select("filepath").eq("doc_id", doc_id).execute()
        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )

        filepath = result.data[0]["filepath"]

        # Get the image from storage
        image_path = f"{filepath}/{page}.png"
        image_data = supabase.storage.from_("doc-images").download(image_path)

        # Convert image to base64
        base64_image = base64.b64encode(image_data).decode('utf-8')

        return {"image": f"data:image/png;base64,{base64_image}"}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/update-redaction-status")
async def update_redaction_status(
    doc_id: str,
    redact_id: str,
    status: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Update the status of a redaction
    """
    token = credentials.credentials
    payload = decode_jwt(token)


    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY, ClientOptions(headers={"Authorization": f"Bearer {token}"}))

        # Update the redaction status
        result = supabase.table("redactions") \
            .update({"status": status, "user_id": payload["sub"], "last_updated": datetime.now().isoformat()}) \
            .eq("doc_id", doc_id) \
            .eq("redact_id", redact_id) \
            .execute()

        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Redaction not found"
            )

        return result.data[0]

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/create-redaction")
async def create_redaction(
    doc_id: str,
    page: int,
    start: int,
    end: int,
    rule: str,
    reasoning: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Create a new redaction entry
    """
    token = credentials.credentials
    decode_jwt(token)

    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY, ClientOptions(headers={"Authorization": f"Bearer {token}"}))

        # Create the redaction
        redaction = {
            "doc_id": doc_id,
            "page": page,
            "start": start,
            "end": end,
            "rule": rule,
            "reasoning": reasoning,
            "status": "Approved",
            "confidence": 1.0
        }

        result = supabase.table("redactions").insert(redaction).execute()

        return result.data[0]

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.delete("/delete-redaction")
async def delete_redaction(
    doc_id: str,
    redact_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Delete a redaction entry
    """
    token = credentials.credentials
    decode_jwt(token)

    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY, ClientOptions(headers={"Authorization": f"Bearer {token}"}))

        # Delete the redaction
        result = supabase.table("redactions") \
            .delete() \
            .eq("doc_id", doc_id) \
            .eq("redact_id", redact_id) \
            .execute()

        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Redaction not found"
            )

        return {"message": "Redaction deleted successfully"}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/invite-user")
async def invite_user(
    doc_id: str,
    email: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Add a user as an editor to a document with pending status
    """
    token = credentials.credentials
    decode_jwt(token)

    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY, ClientOptions(headers={"Authorization": f"Bearer {token}"}))

        # Find user_id from credit table based on email
        credit_result = supabase.table("credits").select("user_id").eq("email", email).execute()

        if not credit_result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        user_id = credit_result.data[0]["user_id"]

        # Create doc_id, user_id pair in docs_users table
        docs_users_entry = {
            "doc_id": doc_id,
            "user_id": user_id,
            "type": "editor",
            "pending": True
        }


        try:
            result = supabase.table("docs_users").insert(docs_users_entry).execute()
        except Exception as e:
            # Check for unique constraint violation
            if hasattr(e, 'args') and e.args and '23505' in str(e.args[0]):
                raise HTTPException(status_code=400, detail="User already exists for document!")
            raise

        print(result)

        return result.data[0]

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/accept-invite")
async def accept_invite(
    doc_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Accept an invite to edit a document
    """
    token = credentials.credentials
    payload = decode_jwt(token)

    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY, ClientOptions(headers={"Authorization": f"Bearer {token}"}))

        # Update the pending status to False
        result = supabase.table("docs_users") \
            .update({"pending": False}) \
            .eq("doc_id", doc_id) \
            .eq("user_id", payload["sub"]) \
            .execute()

        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Invite not found"
            )

        return {"message": "Invite accepted successfully"}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/delete-document-user")
async def delete_document_user(
    doc_id: str,
    user_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Delete a user's access to a document
    """
    token = credentials.credentials
    decode_jwt(token)

    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY, ClientOptions(headers={"Authorization": f"Bearer {token}"}))


        # Delete the user's access to the document
        result = supabase.table("docs_users") \
            .delete() \
            .eq("doc_id", doc_id) \
            .eq("user_id", user_id) \
            .execute()

        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Invite not found"
            )

        return {"message": "User deleted successfully."}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/get-document-users/{doc_id}")
async def get_document_users(
    doc_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Get all users associated with a document
    """
    token = credentials.credentials
    decode_jwt(token)

    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY, ClientOptions(headers={"Authorization": f"Bearer {token}"}))

        # Get all users associated with the document
        result = supabase.table("docs_users") \
            .select("user_id, type, pending") \
            .eq("doc_id", doc_id) \
            .execute()

        #create list of user_ids to get emails from credit table
        user_ids = [user["user_id"] for user in result.data]

        #get user email from credit table
        credit_result = supabase.table("credits") \
            .select("user_id, email") \
            .in_("user_id", user_ids) \
            .execute()

        #create user_id: email dictionary
        credit_dict = {user["user_id"]: user["email"] for user in credit_result.data}

        #merge the results
        for user in result.data:
            user["email"] = credit_dict[user["user_id"]]

        if not result.data:
            return {"users": []}

        return {"users": result.data}

    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/get-pending-invites")
async def get_pending_invites(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Get all pending invites for the authenticated user, including document name and owner email
    """
    token = credentials.credentials
    payload = decode_jwt(token)
    user_id = payload["sub"]

    try:
        # Use service key for metadata queries (bypasses RLS)
        SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
        supabase_service: Client = create_client(SUPABASE_URL, SERVICE_KEY)
        # Get all pending invites for the user
        invites_result = supabase_service.table("docs_users") \
            .select("doc_id") \
            .eq("user_id", user_id) \
            .eq("pending", True) \
            .execute()


        doc_ids = [row["doc_id"] for row in invites_result.data]
        if not doc_ids:
            return {"invites": []}

        # Get document names and owner ids
        docs_result = supabase_service.table("docs") \
            .select("doc_id, filename, owner_id") \
            .in_("doc_id", doc_ids) \
            .execute()


        # Get owner emails
        owner_ids = list({doc["owner_id"] for doc in docs_result.data})
        owners_result = supabase_service.table("credits") \
            .select("user_id, email") \
            .in_("user_id", owner_ids) \
            .execute()

        owner_email_map = {o["user_id"]: o["email"] for o in owners_result.data}

        # Build response
        invites = []
        for doc in docs_result.data:
            invites.append({
                "doc_id": doc["doc_id"],
                "document_name": doc["filename"],
                "owner_email": owner_email_map.get(doc["owner_id"], None)
            })

        return {"invites": invites}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/update-redactions-sensitivity")
async def update_redactions_sensitivity(
    doc_id: str,
    page: int,
    sensitivity: float,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Delete all redactions for a document page with confidence below the given value.
    """
    token = credentials.credentials
    decode_jwt(token)

    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY, ClientOptions(headers={"Authorization": f"Bearer {token}"}))

        # Delete redactions below confidence threshold
        result = supabase.table("redactions") \
            .delete() \
            .eq("doc_id", doc_id) \
            .eq("page", page) \
            .neq("status", "Approved") \
            .lt("confidence", sensitivity) \
            .execute()

        return {"deleted": len(result.data) if result.data else 0}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/export-to-tiff/{doc_id}")
async def export_to_tiff(
    doc_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Export all pages of a document as a downloadable TIFF with bounding boxes for approved redactions.
    """
    import tempfile
    from PIL import Image
    import io

    token = credentials.credentials
    decode_jwt(token)

    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY, ClientOptions(headers={"Authorization": f"Bearer {token}"}))
        # Get file path and page count
        doc_result = supabase.table("docs").select("filepath,page_count").eq("doc_id", doc_id).execute()
        if not doc_result.data:
            raise HTTPException(status_code=404, detail="Document not found")
        filepath = doc_result.data[0]["filepath"]
        page_count = doc_result.data[0]["page_count"]

        # Fetch all approved redactions for this doc
        redactions_result = supabase.table("redactions").select("page,indices").eq("doc_id", doc_id).eq("status", "Approved").execute()
        redactions_by_page = {}
        for r in redactions_result.data:
            redactions_by_page.setdefault(r["page"], []).append(r)

        # Fetch all words for this doc
        words_result = supabase.table("words").select("page,num,x1,y1,x2,y2").eq("doc_id", doc_id).execute()
        words_by_page = {}
        for w in words_result.data:
            words_by_page.setdefault(w["page"], []).append(w)

        images = []
        for page in range(1, page_count + 1):
            image_path = f"{filepath}/{page}.png"
            image_data = supabase.storage.from_("doc-images").download(image_path)
            img = Image.open(io.BytesIO(image_data)).convert("RGBA")
            draw = ImageDraw.Draw(img, "RGBA")

            # Draw black bounding boxes for each word in each approved redaction
            page_redactions = redactions_by_page.get(page, [])
            page_words = words_by_page.get(page, [])
            for redaction in page_redactions:
                indices = redaction.get("indices", [])
                if not indices:
                    continue
                indices = sorted(indices)
                # Split indices into runs of consecutive numbers
                runs = []
                current_run = []
                for idx in indices:
                    if not current_run or idx == current_run[-1] + 1:
                        current_run.append(idx)
                    else:
                        runs.append(current_run)
                        current_run = [idx]
                if current_run:
                    runs.append(current_run)
                # For each run, get the corresponding words and draw a box
                for run in runs:
                    redacted_words = [w for w in page_words if w["num"] in run]
                    redacted_words = sorted(redacted_words, key=lambda w: w["num"])
                    # Group words into lines based on x1 and y1/y2 (existing logic)
                    lines = []
                    current_line = []
                    prev_word = None
                    width, height = img.size
                    x_threshold = 0.03 * width  # 3% of width
                    for w in redacted_words:
                        if prev_word is not None:
                            if (w["x1"] * width < prev_word["x1"] * width - x_threshold) and (w["y1"] * height > prev_word["y2"] * height):
                                if current_line:
                                    lines.append(current_line)
                                    current_line = []
                        current_line.append(w)
                        prev_word = w
                    if current_line:
                        lines.append(current_line)
                    # Draw one black box per line
                    for line_words in lines:
                        if not line_words:
                            continue
                        x1 = min(w["x1"] for w in line_words)
                        y1 = min(w["y1"] for w in line_words)
                        x2 = max(w["x2"] for w in line_words)
                        y2 = max(w["y2"] for w in line_words)
                        width, height = img.size
                        box_height = y2 - y1
                        expand = 0.025 * box_height
                        y1_expanded = max(0, y1 - expand)
                        y2_expanded = min(1, y2 + expand)
                        box = [x1 * width, y1_expanded * height, x2 * width, y2_expanded * height]
                        draw.rectangle(box, outline=(0,0,0,255), width=2, fill=(0,0,0,255))
            images.append(img.convert("RGB"))

        # Save all images as a single multi-page TIFF
        with tempfile.NamedTemporaryFile(suffix=".tiff", delete=False) as tmp:
            images[0].save(tmp.name, save_all=True, append_images=images[1:], format="TIFF")
            # images[0].save("test.tiff", save_all=True, append_images=images[1:], format="TIFF")
            tmp.seek(0)
            tiff_bytes = tmp.read()
        filename = f"{doc_id}_redacted.tiff"
        # logger.info(f"TIFF size: {len(tiff_bytes)} bytes")
        return StreamingResponse(io.BytesIO(tiff_bytes), media_type="image/tiff", headers={"Content-Disposition": f"attachment; filename={filename}"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-all-stats")
async def get_all_stats(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Get combined document statistics and chart statistics for the authenticated user.
    Returns both overall statistics and chart data in a single call.

    The returned data is a JSON object with two keys: "stats" and "chart_stats".
    The "stats" object contains the following keys:
    - total_documents: The total number of documents
    - total_pages: The total number of pages
    - declassifiable_pages: The total number of pages minus redacted pages
    - declassifiable_percentage: The percentage of words that are not redacted

    The "chart_stats" object contains the following keys:
    - declassifiable_percentage: The percentage of words that are not redacted
    - documents: A list of objects containing document names and their respective
      declassifiable and classified percentages
    """
    token = credentials.credentials
    payload = decode_jwt(token)

    try:
        supabase: Client = create_client(
            SUPABASE_URL,
            SUPABASE_KEY,
            ClientOptions(headers={"Authorization": f"Bearer {token}"})
        )
        # Get all documents that aren't loading
        docs_result = supabase.table("docs").select("*").neq("status", "Error").order("created_at", desc=True).execute()
        total_documents = len(docs_result.data)

        if total_documents == 0:
            return {
                "stats": {
                    "total_documents": 0,
                    "total_pages": 0,
                    "declassifiable_pages": 0,
                    "declassifiable_percentage": 0
                },
                "chart_stats": {
                    "declassifiable_percentage": 0,
                    "documents": []
                }
            }


        total_pages = sum(doc["page_count"] for doc in docs_result.data)
        total_words = sum(doc["word_count"] for doc in docs_result.data)

        # Get all redactions
        all_redactions = supabase.table("redactions").select("*").execute()

        # Calculate redacted pages

        redacted_pages = len(set((r['doc_id'], r['page']) for r in all_redactions.data))
        declassifiable_pages = total_pages - redacted_pages



        # Calculate redacted words per document
        redacted_words_per_doc = {}
        for r in all_redactions.data:
            if not r.get('indices'):  # Skip if indices is None or empty
                continue
            doc_id = r.get('doc_id')
            if not doc_id:  # Skip if doc_id is missing
                continue
            if doc_id not in redacted_words_per_doc:
                redacted_words_per_doc[doc_id] = 0
            redacted_words_per_doc[doc_id] += len(r['indices'])

        total_redacted_words = sum(redacted_words_per_doc.values())


        # Calculate overall declassifiable percentage
        declassifiable_percentage = 0 if total_words == 0 else round(
            ((total_words - total_redacted_words) / total_words) * 100,
            2
        )

        # Process recent documents for chart data
        recent_documents = []
        for doc in docs_result.data[:5]:
            doc_total = doc["word_count"]
            doc_redacted = redacted_words_per_doc.get(doc['doc_id'], 0)

            doc_declassifiable = 0 if doc_total == 0 else round(
                ((doc_total - doc_redacted) / doc_total) * 100,
                2
            )
            doc_classified = 100 - doc_declassifiable

            recent_documents.append({
                "name": doc["filename"],
                "declassifiable": doc_declassifiable,
                "classified": doc_classified
            })

        return {
            "stats": {
                "total_documents": total_documents,
                "total_pages": total_pages,
                "declassifiable_pages": declassifiable_pages,
                "declassifiable_percentage": declassifiable_percentage
            },
            "chart_stats": {
                "declassifiable_percentage": declassifiable_percentage,
                "documents": recent_documents
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.delete("/delete-document/{doc_id}")
async def delete_document(
    doc_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Delete a document and its associated files from storage.
    Database records will be cleaned up by triggers.
    """
    token = credentials.credentials
    decode_jwt(token)

    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY, ClientOptions(headers={"Authorization": f"Bearer {token}"}))

        # Get the filepath from the docs table
        result = supabase.table("docs").select("filepath").eq("doc_id", doc_id).execute()

        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )

        filepath = result.data[0]["filepath"]
        filepath_pdf = f"{filepath}.pdf"

        try:
            # Delete PDF from docs storage
            supabase.storage.from_("docs").remove([filepath_pdf])

            # Delete all page images from doc-images storage
            # Note: This assumes images are named 1.png, 2.png, etc.
            page_count_result = supabase.table("docs").select("page_count").eq("doc_id", doc_id).execute()
            if page_count_result.data:
                page_count = page_count_result.data[0]["page_count"]
                image_paths = [f"{filepath}/{i+1}.png" for i in range(page_count)]
                supabase.storage.from_("doc-images").remove(image_paths)

            return {"message": "Document deleted successfully"}

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete document files: {str(e)}"
            )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.delete("/delete-guideline/{guideline_id}")
async def delete_guideline(
    guideline_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Delete a guideline document from storage.
    Database records will be cleaned up by triggers.
    """
    token = credentials.credentials
    decode_jwt(token)

    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY, ClientOptions(headers={"Authorization": f"Bearer {token}"}))

        # Get the filepath from the guidelines table
        result = supabase.table("guidelines").select("filepath").eq("guideline_id", guideline_id).execute()

        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Guideline not found"
            )

        filepath = result.data[0]["filepath"]
        filepath_pdf = f"{filepath}.pdf"

        try:
            # Delete PDF from guidelines storage
            supabase.storage.from_("guidelines").remove([filepath_pdf])
            return {"message": "Guideline deleted successfully"}

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete guideline file: {str(e)}"
            )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    
class RedactionIndicesUpdateBody(BaseModel):
    redact_id: str
    indices: List[int]

@app.post("/update-redactions-indices")
async def update_redactions_indices(
    update: RedactionIndicesUpdateBody,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Update the indices of a redaction. Expects JSON body with 'redact_id' and 'indices'.
    """
    token = credentials.credentials
    decode_jwt(token)
    try:
        supabase: Client = create_client(
            SUPABASE_URL,
            SUPABASE_KEY,
            ClientOptions(headers={"Authorization": f"Bearer {token}"})
        )
        result = supabase.table("redactions") \
            .update({"indices": update.indices, "last_updated": datetime.now().isoformat()}) \
            .eq("redact_id", update.redact_id) \
            .execute()
        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Redaction not found"
            )
        return result.data[0]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/get-guideline-rules/{guideline_id}")
async def get_guideline_rules(
    guideline_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Get all rules associated with a specific guideline ID
    """
    token = credentials.credentials
    decode_jwt(token)

    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY, ClientOptions(headers={"Authorization": f"Bearer {token}"}))

        # Get all rules for this guideline
        rules_result = supabase.table("declass_rules").select("*").eq("guideline_id", guideline_id).execute()

        if not rules_result.data:
            return {"rules": []}


        return {"rules": rules_result.data}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/edit-guideline-rule")
async def edit_guideline_rule(
    guideline_id: str,
    rule: str,
    field: str,
    value: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Edit a specific field of a guideline rule.

    Args:
        guideline_id: The ID of the guideline
        rule: The rule identifier (e.g., "3.3.1-28")
        field: The field to update (must be one of: grouping, level, content, duration, reason, remarks)
        value: The new value for the field
    """
    token = credentials.credentials
    decode_jwt(token)

    # Validate field name
    valid_fields = {"grouping", "level", "content", "duration", "reason", "remarks"}
    if field not in valid_fields:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid field name. Must be one of: {', '.join(valid_fields)}"
        )

    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY, ClientOptions(headers={"Authorization": f"Bearer {token}"}))

        # Check if the rule exists
        rule_result = supabase.table("declass_rules") \
            .select("*") \
            .eq("guideline_id", guideline_id) \
            .eq("rule", rule) \
            .execute()

        if not rule_result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Rule not found"
            )

        # Update the specified field
        result = supabase.table("declass_rules") \
            .update({field: value}) \
            .eq("guideline_id", guideline_id) \
            .eq("rule", rule) \
            .execute()

        return result.data[0]

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
