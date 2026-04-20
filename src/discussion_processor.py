import os
import time

import ollama

DISCUSSION_DIR = "discussions"
TRIMMED_DIR = f"{DISCUSSION_DIR}/trimmed"
OUTPUT_DIR = f"{DISCUSSION_DIR}/out"

def simplify_discussion(discussion_text):
    start = time.time()
    extraction_response = ollama.chat(
        model='llama3.1:8b-instruct-q4_K_M',
        messages=[
            {
                'role': 'system',
                'content': """Extract every meteorologically significant claim 
                from the following forecast discussion as a bullet list. 
                Quote directly from the text where possible, and do not add any 
                information not present in the text. Only include the bullet list
                in your response."""
            },
            {
                'role': 'user',
                'content': discussion_text
            }
        ]
    )
    extracted_claims = extraction_response['message']['content']
    print(f"{extracted_claims}\n")

    response = ollama.chat(
        model='llama3.1:8b-instruct-q4_K_M',
        messages=[
            {
                'role': 'system',
                'content': """You are a meteorologist providing a weather forecast 
                for a general audience. Translate the following meteorological claims into 
                plain language for a general audience, providing a single summary for the 
                entire forecast period. Do not add any information beyond what is listed.
                
                Your output must follow this exact format:
                PATTERN: 2-3 sentences describing the large-scale synoptic weather pattern
                IMPACTS: 4-5 sentences describing what this means for local weather
                CONFIDENCE: Low, medium, or high
                """
            },
            {
                'role': 'user',
                'content': f"""Translate these claims:\n\n{extracted_claims}. 
                Only include the simplified text in your response."""
            }
        ]
    )

    end = time.time()
    # print(f"Total time: {end - start}")
    return response['message']['content'], end - start

for filename in os.listdir(TRIMMED_DIR):
    with open(f"{TRIMMED_DIR}/{filename}", "r") as f:
        print(f"Processing discussion {filename}")
        discussion, time_taken = f.read()
        print(f"Processed discussion {filename} in {time_taken} seconds")

    simplified = simplify_discussion(discussion)
    with open(f"{OUTPUT_DIR}/{filename}_s", "w") as out_file:
        out_file.write(discussion)
