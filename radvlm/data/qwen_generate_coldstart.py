#!/usr/bin/env python
# run_coldstart_parallel_with_shuffle_and_skip.py
import argparse
import base64
import io
import json
import pathlib
import re
import sys
import random
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image, ImageOps
from openai import OpenAI

def parse_args():
    p = argparse.ArgumentParser(
        description="Run cold-start inference in parallel on a shuffled dataset and save each output as individual JSON files, skipping existing ones."
    )
    p.add_argument(
        "--json", required=True,
        help="Input dataset in JSON format"
    )
    p.add_argument(
        "--out_dir", required=True,
        help="Directory where per-example outputs will be saved"
    )
    p.add_argument(
        "--max_tokens", type=int, default=3000,
        help="Maximum tokens for the model reply"
    )
    p.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel worker processes"
    )
    return p.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Configuration: adjust as needed
# ─────────────────────────────────────────────────────────────────────────────
CLIENT = OpenAI(
    api_key="sk-rc-NagDM-aFcL2ssAn03MyXMQ",
    base_url="https://api.swissai.cscs.ch/v1",
)
MAX_SIDE = 512
JPEG_QUAL = 90
TEMPLATE_PATH = "/capstor/scratch/cscs/ndeperr/code/RadVLM-r1/RadVLM/radvlm/data/prompt_coldstart.txt"
template_text = pathlib.Path(TEMPLATE_PATH).read_text(encoding='utf-8')
THINK_SUFFIX = (
    " First output the thinking process in <think> </think> tags and then "
    "output the final answer in <answer> </answer> tags."
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def fill_template(question: str, answer_gt: str) -> str:
    prompt = (
        template_text
        .replace("{QUESTION}", question)
        .replace("{ANSWER_GT}", answer_gt)
    )
    return re.sub(r".*{IMAGE}.*\n?", "", prompt).strip()

def data_uri_from_image(path: str) -> str:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        if max(im.size) > MAX_SIDE:
            im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=JPEG_QUAL)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"

# ─────────────────────────────────────────────────────────────────────────────
# Worker function
# ─────────────────────────────────────────────────────────────────────────────
def process_entry(entry: dict, out_dir: pathlib.Path, max_tokens: int) -> None:
    """
    Process a single dataset entry, call the model, and write output JSON per example,
    skipping if the output file already exists.
    """
    img_path = entry['image']
    # Determine output file path early to skip if existing
    identifier = entry.get('id','') or pathlib.Path(img_path).stem
    out_file = out_dir / f"{identifier}.json"
    if out_file.exists():
        print(f"→ Skipping {identifier}, already exists at {out_file}")
        return

    # Extract human question and GPT ground truth
    convs     = entry['conversations']
    human_val = next(c for c in convs if c['from']=='human')['value']
    gpt_gt    = next(c for c in convs if c['from']=='gpt')['value']
    orig_q    = human_val.split("\n",1)[1].strip()
    prompt_model = fill_template(orig_q, gpt_gt)
    
    # Build message with image
    data_uri = data_uri_from_image(img_path)
    messages = [{
        'role': 'user',
        'content': [
            {'type':'image_url','image_url':{'url':data_uri}},
            {'type':'text','text':prompt_model},
        ]
    }]
    
    # Call model
    reply = CLIENT.chat.completions.create(
        model='Qwen/Qwen2.5-VL-72B-Instruct',
        messages=messages,
        max_tokens=max_tokens
    ).choices[0].message.content
    
    # Prepare output dictionary
    result = {
        'image': img_path,
        'conversations': [
            {'from':'human', 'value':f"<image>\n{orig_q}{THINK_SUFFIX}"},
            {'from':'gpt',   'value':reply},
        ],
        'id':     entry.get('id',''),
        'labels': entry.get('labels',[]),
    }
    # Write single JSON file
    with out_file.open('w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    print(f"✓ Saved {identifier} -> {out_file}")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    src = pathlib.Path(args.json)
    out_d = pathlib.Path(args.out_dir)
    if not out_d.exists():
        out_d.mkdir(parents=True)

    # Load and shuffle dataset to randomize processing order
    dataset = json.loads(src.read_text(encoding='utf-8'))
    random.shuffle(dataset)
    print(f"Loaded and shuffled {len(dataset)} examples from {src}")

    # Partial worker
    worker = partial(
        process_entry,
        out_dir=out_d,
        max_tokens=args.max_tokens,
    )

    # Parallel execution
    with ProcessPoolExecutor(max_workers=args.workers) as exe:
        futures = {exe.submit(worker, entry): entry for entry in dataset}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                entry = futures[fut]
                sys.stderr.write(f"Error processing {entry.get('id','?')}: {e}\n")

if __name__ == '__main__':
    main()
