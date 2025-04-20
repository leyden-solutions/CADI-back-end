import os
import json
import uuid
import pathlib
from google import genai
from google.genai import types
from dotenv import load_dotenv

from prompt import guidelines_prompt,sections_prompt, example_response
from supabase import create_client, Client, ClientOptions

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_KEY,
    ClientOptions(headers={"Authorization": f"Bearer {"eyJhbGciOiJIUzI1NiIsImtpZCI6Ilh1NzQ5ZWRadUFzaG56RkkiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL3Vwd3hqZGVmb29vZHptbWZqYnp4LnN1cGFiYXNlLmNvL2F1dGgvdjEiLCJzdWIiOiI5Y2M4ZDExYy0zYmFhLTQ2ZTctYjBjZS02NGVkMTM3ODJjMmUiLCJhdWQiOiJhdXRoZW50aWNhdGVkIiwiZXhwIjoxNzQ1MDMxNDU4LCJpYXQiOjE3NDUwMjc4NTgsImVtYWlsIjoiYWxleGFuZGVyaGFtaWRpMUBnbWFpbC5jb20iLCJwaG9uZSI6IiIsImFwcF9tZXRhZGF0YSI6eyJwcm92aWRlciI6ImVtYWlsIiwicHJvdmlkZXJzIjpbImVtYWlsIl19LCJ1c2VyX21ldGFkYXRhIjp7ImVtYWlsIjoiYWxleGFuZGVyaGFtaWRpMUBnbWFpbC5jb20iLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwicGhvbmVfdmVyaWZpZWQiOmZhbHNlLCJzdWIiOiI5Y2M4ZDExYy0zYmFhLTQ2ZTctYjBjZS02NGVkMTM3ODJjMmUifSwicm9sZSI6ImF1dGhlbnRpY2F0ZWQiLCJhYWwiOiJhYWwxIiwiYW1yIjpbeyJtZXRob2QiOiJvdHAiLCJ0aW1lc3RhbXAiOjE3NDQ3ODM2MzJ9XSwic2Vzc2lvbl9pZCI6Ijk1NmYxNTgyLWQ2ZTgtNDk0My05YTllLTQ3YzdiNjBiYTczZiIsImlzX2Fub255bW91cyI6ZmFsc2V9.E8DsEasZyteYf3Ty2jSGY0n06h2eCxSuaSA2prJLR5w"}"})
)

user_id = "9cc8d11c-3baa-46e7-b0ce-64ed13782c2e"
guideline_id = str(uuid.uuid4())

def add_to_table(guideline_data):


    guideline_id = "3d5f17ef-f7bb-44be-ac45-4f438fc2dcbd"


    supabase.table("guidelines").insert({
        "guideline_id": guideline_id,
        "owner_id": user_id,
        "filepath":"who cares",
        "filename":"who cares"
    }).execute()


    grouping = next(iter(guideline_data))
    for i, item in enumerate(guideline_data[grouping]["items"]):
        rule = f"{grouping}-{i+1}"
        print("""inserting into table:
rule: {rule}
grouping: {grouping}
content: {content}
level: {level}
duration: {duration}
reason: {reason}
remarks: {remarks}
\n\n""".format(rule=rule, grouping=grouping, content=item["content"], level=item["level"], duration=item["duration"], reason=item["reason"], remarks=item["remarks"]))


        supabase.table("declass_rules").insert({
            "guideline_id": guideline_id,
            "rule": rule,
            "grouping": grouping,
            "content": item["content"],
            "level": item["level"],
            "duration": item["duration"],
            "reason": item["reason"],
            "remarks": item["remarks"]
        }).execute()



def process_pdf_with_gemini(pdf_path: str, sections: str):
    """
    Process a PDF file using Gemini's model and return the generated content.
    """



    client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

    # Convert the PDF path to a pathlib.Path object
    filepath = pathlib.Path(pdf_path)


    prompt = guidelines_prompt.replace("{sections}", sections)


    # Generate content using Gemini model
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            types.Part.from_bytes(
                data=filepath.read_bytes(),
                mime_type='application/pdf',
            ),
            prompt
        ]
    )
    text = response.text
    text = text.replace("```json", "").replace("```", "")

    return json.loads(text)

def get_sections_from_pdf(pdf_path: str):

    client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    filepath = pathlib.Path(pdf_path)
    response = client.models.generate_content(
        model="gemini-2.5-pro-exp-03-25",
        contents=[
            types.Part.from_bytes(
                data=filepath.read_bytes(),
                mime_type='application/pdf',
            ),
            sections_prompt
        ]
    )
    text = response.text
    text = text.replace("```json", "").replace("```", "")
    return json.loads(text)['sections']

def main():

    original_filename = "who cares"
    filepath = f"hello-{guideline_id}"
    batch_size = 3
    # Initialize Gemini client

    # Path to your PDF
    pdf_path = 'assets/default_rules_gd.pdf'

    sections = get_sections_from_pdf(pdf_path)
    print(sections)


    rules_data = []

    for i in range(0, len(sections), batch_size):
        print(f"Processing batch {i//batch_size + 1} of {len(sections)//batch_size}")
        batch_sections = sections[i:i+batch_size]

        section_str = f"[{', '.join(batch_sections)}]"
        rules_dict = process_pdf_with_gemini(pdf_path, section_str)
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

    supabase.table("declass_rules").insert(rules_data).execute()
    guideline_data = {
        "guideline_id": guideline_id,
        "filename": original_filename,
        "filepath": filepath,
        "status": "Ready"
    }
    supabase.table("guidelines").insert(guideline_data).execute()
    print(f"Inserted {len(rules_data)} rules for batch {i//batch_size + 1}")

    # Moved outside both loops to insert all rules at once

    # Process the PDF and get the summary
    # sections = get_sections_from_pdf(pdf_path)
    # sections = ['3.1.1', '3.1.2', '3.1.3', '3.1.4', '3.1.5', '3.1.6', '3.2', '3.3', '3.3.1', '3.4.1', '3.4.2', '3.4.3', '3.4.4', '3.5.1', '3.5.2', '3.5.3', '3.6.1', '3.6.2', '3.7.1', '3.7.2', '3.7.3', '3.7.4', '3.7.5', '3.8']
    # print(sections)
    # for section in sections:
    #     summary = process_pdf_with_gemini(pdf_path, section)
    #     print(summary)
    #     print(f"Section: {section}\n\n")
    #     # print(json.dumps(summary, indent=4))
    #     for item in summary.values():
    #         print(f"{item}\n\n")

    # add_to_table(summary)
    # sections = get_sections_from_pdf(pdf_path)
    # print(sections)


if __name__ == "__main__":
    main()
