import os
import time
import json
from pathlib import Path

from transformers import pipeline
import torch

from discussion_retrieval import process_zip

DISCUSSION_DIR = "discussions"
TRIMMED_DIR = f"{DISCUSSION_DIR}/trimmed"
OUTPUT_DIR = f"{DISCUSSION_DIR}/out"

pipe = pipeline(
    "text-generation",
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    device="mps",
    torch_dtype=torch.float16
)

def batch_extract(discussions, batch_size=4):
    discussion_texts = []

    for filename in discussions:
        with open(f"{TRIMMED_DIR}/{filename}", "r") as f:
            discussion = f.read()

        discussion_texts.append(discussion)

    print("Extracting claims...")
    extraction_prompts = [
        f"""<|system|>
Extract every meteorologically significant claim from the following forecast 
discussion as a bullet list. Quote directly from the text where possible, and 
do not add any information not present in the text. Only include the bullet list
in your response.
<|user|>
{text}
<|assistant|>"""
        for text in discussion_texts
    ]

    extracted_claims = []

    start = time.time()
    for i in range(0, len(extraction_prompts), batch_size):
        batch = extraction_prompts[i:i + batch_size]
        print(f"  Extraction batch {i//batch_size + 1}/"
              f"{len(extraction_prompts)//batch_size + 1}")

        outputs = pipe(
            batch,
            max_new_tokens=300,
            do_sample=False,
            batch_size=batch_size,
            pad_token_id=pipe.tokenizer.eos_token_id
        )

        for output in outputs:
            claims = output[0]['generated_text'].split('<|assistant|>')[-1].strip()
            extracted_claims.append(claims)

    end = time.time()

    extraction_checkpoint = f"{DISCUSSION_DIR}/extracted_claims.json"
    with open(extraction_checkpoint, "w") as f:
        json_extraction_data = [
            {"filename": filename, "original": text, "claims": claims}
            for filename, text, claims in zip(discussions, discussion_texts, extracted_claims)
        ]

        json.dump(json_extraction_data, f, indent=2)

    print(f"Extraction complete in {end - start} seconds ({(end - start) / len(extraction_prompts)} per discussion)")
    print(f"Checkpoint saved to {extraction_checkpoint}")
    return [d["claims"] for d in json_extraction_data]

def batch_simplify(discussions, batch_size=8, extraction_batch_size=4):
    extraction_checkpoint = f"{DISCUSSION_DIR}/extracted_claims.json"
    if Path(extraction_checkpoint).is_file():
        print("Found extraction checkpoint")
        with open(extraction_checkpoint) as f:
            checkpoint_data = json.load(f)
        
        extracted_claims = [d["claims"] for d in checkpoint_data]
    else:
        print("No checkpoint found.")
        extracted_claims = batch_extract(discussions, batch_size=extraction_batch_size)

    simplification_prompts = [
        f"""<|system|>
You are a meteorologist providing a weather forecast for a general audience. 
Translate the following meteorological claims into plain language for a 
general audience using the format below. Do not add any information beyond 
what is listed.

Your output must follow this exact format:
PATTERN: 2-3 sentences describing the large-scale synoptic weather pattern
IMPACTS: 4-5 sentences describing what this means for local weather
<|user|>
Translate these claims:
{claims}
<|assistant|>"""
        for claims in extracted_claims
    ]

    print("Simplifying extracted claims")

    results = []
    for i in range(0, len(simplification_prompts), batch_size):
        batch_prompts = simplification_prompts[i:i + batch_size]
        batch_meta = discussions[i:i + batch_size]
        batch_claims = extracted_claims[i:i + batch_size]

        outputs = pipe(
            batch_prompts,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            batch_size=batch_size,
            pad_token_id=pipe.tokenizer.eos_token_id
        )

        for filename, claims, output in zip(batch_meta, batch_claims, outputs):
            simplified = output[0]['generated_text'].split('<|assistant|>')[-1].strip()
            results.append({
                "filename": filename,
                "extracted_claims": claims,
                "simplified": simplified,
            })

        # Incremental save after each batch
        checkpoint_path = f"{DISCUSSION_DIR}.results_checkpoint.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(results, f, indent=2)

    return results

if __name__ == "__main__":
    process_zip("2025-07-01T00:00Z", "2026-04-25T23:59Z")
    batch_simplify(os.listdir(TRIMMED_DIR))
