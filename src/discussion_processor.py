import os
import re
import multiprocessing as mp
import psutil
import subprocess
from functools import partial
import time
from pathlib import Path

from llama_cpp import Llama

from discussion_retrieval import process_zip

DISCUSSION_DIR = "discussions"
TRIMMED_DIR = f"{DISCUSSION_DIR}/trimmed"
OUTPUT_DIR = f"{DISCUSSION_DIR}/out"

MODEL_PATH = "llama/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

def has_gpu():
    try:
        return subprocess.run(['nvidia-smi'], capture_output=True).returncode == 0
    except FileNotFoundError:
        return False

def simplify_discussion(llm, discussion_text):
    extraction_response = llm.create_chat_completion(
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
    extracted_claims = extraction_response['choices'][0]['message']['content']
    # print(f"{extracted_claims}\n")

    response = llm.create_chat_completion(
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
    return response['choices'][0]['message']['content']

def validate_output(text):
    errors = []

    for section in ('PATTERN', 'IMPACTS', 'CONFIDENCE'):
        count = len(re.findall(rf'^{section}:', text, re.MULTILINE))
        if count == 0:
            errors.append(f"missing {section} section")
        elif count > 1:
            errors.append(f"multiple {section} sections ({count}) — output may contain more than one summary")

    if not errors:
        pattern_pos = text.index('PATTERN:')
        impacts_pos = text.index('IMPACTS:')
        confidence_pos = text.index('CONFIDENCE:')
        if not (pattern_pos < impacts_pos < confidence_pos):
            errors.append("sections out of order (expected PATTERN → IMPACTS → CONFIDENCE)")

        match = re.search(r'^CONFIDENCE:\s*(\w+)', text, re.MULTILINE)
        if match and match.group(1).lower() not in ('low', 'medium', 'high'):
            errors.append(f"invalid CONFIDENCE value '{match.group(1)}' (expected Low, Medium, or High)")

    return errors

def worker_process(discussion_chunk, model_path=MODEL_PATH, n_threads=1, n_gpu_layers=0):
    llm = Llama(model_path=model_path, n_threads=n_threads, n_gpu_layers=n_gpu_layers, n_ctx=8192, verbose=False)
    for filename in discussion_chunk:
        print(f"Processing discussion {filename}")
        with open(f"{TRIMMED_DIR}/{filename}", "r") as f:
            discussion = f.read()

        try:
            result = simplify_discussion(llm, discussion)
            errors = validate_output(result)
            if errors:
                print(f"Warning: output for {filename} failed validation: {'; '.join(errors)}")
            out_filename = Path(f"{OUTPUT_DIR}/{filename}_s")
            out_filename.parent.mkdir(exist_ok=True, parents=True)
            with open(out_filename, "w") as out_file:
                out_file.write(result)
        except Exception as e:
            print(f"Encountered the following exception when processing discussion {filename}: {e} ")

if __name__ == "__main__":
    process_zip("2026-04-01T00:00Z", "2026-04-30T23:59Z")

    discussion_filenames = os.listdir(TRIMMED_DIR)
    start = time.time()

    if has_gpu():
        print("GPU detected — using sequential processing")
        worker_process(discussion_filenames, model_path=MODEL_PATH, n_gpu_layers=-1)
    else:
        print("No GPU detected — using multiprocessing")
        n_physical_cores = psutil.cpu_count(logical=False)
        n_workers = max(1, n_physical_cores - 1)
        n_threads = max(1, n_physical_cores // n_workers)
        chunks = [discussion_filenames[i::n_workers] for i in range(n_workers)]
        worker_fn = partial(worker_process, model_path=MODEL_PATH, n_threads=n_threads)
        with mp.Pool(processes=n_workers) as pool:
            pool.map(worker_fn, chunks)

    end = time.time()
    print(f"Processed discussions in {end - start} seconds ({(end - start) / len(discussion_filenames)} seconds per discussion)")
