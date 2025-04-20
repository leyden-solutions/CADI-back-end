import os
import io
import tempfile
import gc
from pdf2image import convert_from_path
from PIL import Image, ImageDraw
import pytesseract
import sys
from PyPDF2 import PdfReader, PdfWriter

# Standalone process_pdf logic copied from app.py

def process_pdf(pdf_bytes, doc_id=None):
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
        finally:
            os.unlink(temp_pdf_path)
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        raise
    return {"page_count": page_count, "words": results}

SAMPLE_PDF_PATH = os.path.join(os.path.dirname(__file__), '../test_data/jktest.pdf')
RESULT_PDF_PATH = os.path.join(os.path.dirname(__file__), 'result.pdf')
RESULT_BOXES_PDF_PATH = os.path.join(os.path.dirname(__file__), 'result_with_boxes.pdf')


def draw_boxes_on_pdf(input_pdf_path, ocr_results, output_pdf_path):
    # Map from 1-based page number to list of boxes
    page_word_map = {}
    for word in ocr_results['words']:
        page_word_map.setdefault(word['page'], []).append(word)

    # Convert PDF pages to images, draw boxes, and save as new PDF
    images = convert_from_path(input_pdf_path, dpi=300)
    boxed_images = []
    for idx, img in enumerate(images):
        page_num = idx + 1
        draw = ImageDraw.Draw(img)
        width, height = img.size
        for word in page_word_map.get(page_num, []):
            x1 = int(word['x1'] * width)
            y1 = int(word['y1'] * height)
            x2 = int(word['x2'] * width)
            y2 = int(word['y2'] * height)
            draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
        boxed_images.append(img.convert('RGB'))
    # Save all boxed images as a PDF
    if boxed_images:
        boxed_images[0].save(output_pdf_path, save_all=True, append_images=boxed_images[1:])


def main():
    if not os.path.exists(SAMPLE_PDF_PATH):
        print(f"Sample PDF not found at {SAMPLE_PDF_PATH}")
        return
    with open(SAMPLE_PDF_PATH, 'rb') as f:
        pdf_bytes = f.read()
    # Output the PDF bytes to result.pdf
    with open(RESULT_PDF_PATH, 'wb') as out_f:
        out_f.write(pdf_bytes)
    result = process_pdf(pdf_bytes)
    print(f"OCR extracted {len(result['words'])} words from {result['page_count']} pages.")
    print(f"First 5 words: {result['words'][:5]}")
    # Draw bounding boxes and save as PDF
    draw_boxes_on_pdf(SAMPLE_PDF_PATH, result, RESULT_BOXES_PDF_PATH)
    print(f"PDF with bounding boxes saved to {RESULT_BOXES_PDF_PATH}")

if __name__ == "__main__":
    main()