import os
import multiprocessing as mp
from functools import partial
import time
from pathlib import Path

import ollama

DISCUSSION_DIR = "discussions"
TRIMMED_DIR = f"{DISCUSSION_DIR}/trimmed"
OUTPUT_DIR = f"{DISCUSSION_DIR}/out"

def simplify_discussion(discussion_text, model="llama3.1:8b-instruct-q4_K_M"):
    extraction_response = ollama.chat(
        model=model,
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
    # print(f"{extracted_claims}\n")

    response = ollama.chat(
        model=model,
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

    # print(f"Total time: {end - start}")
    return response['message']['content']

def worker_process(discussion_chunk, model="llama3.1:8b-instruct-q4_K_M"):
    for filename in discussion_chunk:
        print(f"Processing discussion {filename}")
        with open(f"{TRIMMED_DIR}/{filename}", "r") as f:
            discussion = f.read()

        try:
            result = simplify_discussion(discussion, model)
            out_filename = Path(f"{OUTPUT_DIR}/{filename}_s")
            out_filename.parent.mkdir(exist_ok=True, parents=True)
            with open(out_filename, "w") as out_file:
                out_file.write(result)
        except Exception as e:
            print(f"Encountered the following exception when processing discussion {filename}: {e} ")

if __name__ == "__main__":
    n_workers = max(1, mp.cpu_count() - 1)
    discussion_filenames = os.listdir(TRIMMED_DIR)
    chunks = [discussion_filenames[i::n_workers] for i in range(n_workers)]
    worker_fn = partial(worker_process)

    start = time.time()
    with mp.Pool(processes=n_workers) as pool:
        pool.map(worker_fn, chunks)
    end = time.time()

    print(f"Processed discussions in {end - start} seconds ({(end - start) / len(discussion_filenames)} seconds per discussion)")
