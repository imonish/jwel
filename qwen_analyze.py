#!/usr/bin/env python3

"""
CPU-only jewellery analysis using local Qwen/Qwen3.5-2B.

Input:
    data/output/
        ├── BiRefNet/
        ├── InSPyReNet/
        └── IS-Net/

Output:
    JSON files are saved beside the corresponding PNG image.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

# ============================================================
# CPU ONLY
# ============================================================

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen3_5ForConditionalGeneration,
)

# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_DIR = (
    PROJECT_DIR
    / "models"
    / "Qwen3.5-2B"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "output"
)

MODEL_NAME = "Qwen/Qwen3.5-2B"

# Process these folders
MODEL_OUTPUTS = [
    "BiRefNet",
    "InSPyReNet",
    "IS-Net",
]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}

MAX_NEW_TOKENS = 512


# ============================================================
# PROMPT
# ============================================================

def build_prompt(filename: str) -> str:

    return f"""
You are a jewellery image analysis system.

Analyze the jewellery shown in this image.

Return ONLY valid JSON.
Do not use Markdown.
Do not add explanations outside the JSON.

Use exactly this structure:

{{
  "image": "{filename}",
  "jewellery_type": "",
  "metal": "",
  "metal_color": "",
  "gemstones": [],
  "gemstone_count": 0,
  "design": "",
  "chain_present": false,
  "approximate_objects": []
}}

Rules:

1. Identify the main jewellery type.
2. Identify the apparent metal.
3. Identify the metal color.
4. List visible gemstones if identifiable.
5. Count visible gemstones only when reasonably possible.
6. Describe the design briefly.
7. Set chain_present to true only if a chain is visible.
8. approximate_objects should contain the visible jewellery/object categories.
9. Do not invent information.
10. If something cannot be determined, use "unknown".
11. gemstone_count must be an integer.
12. chain_present must be true or false.
"""


# ============================================================
# FIND IMAGES
# ============================================================

def find_images():

    images = []

    for folder_name in MODEL_OUTPUTS:

        folder = OUTPUT_DIR / folder_name

        if not folder.exists():

            print(
                f"[WARNING] Folder not found: {folder}"
            )

            continue

        for path in sorted(folder.iterdir()):

            if (
                path.is_file()
                and path.suffix.lower()
                in IMAGE_EXTENSIONS
            ):

                images.append(path)

    return images


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("=" * 60)
    print("Qwen3.5-2B Jewellery Analysis")
    print("=" * 60)

    print(f"Model: {MODEL_NAME}")
    print(f"Model directory: {MODEL_DIR}")
    print("Device: CPU")
    print("Dtype: float32")

    if not MODEL_DIR.exists():

        raise FileNotFoundError(
            f"Model directory does not exist:\n{MODEL_DIR}"
        )

    print("\nLoading processor...")

    processor = AutoProcessor.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )

    print("Loading Qwen3.5-2B...")

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_DIR,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )

    model.to("cpu")
    model.eval()

    print("Model loaded successfully!")

    return processor, model


# ============================================================
# EXTRACT JSON
# ============================================================

def extract_json(text: str):

    text = text.strip()

    # Remove Markdown code fences if model adds them
    if text.startswith("```"):

        lines = text.splitlines()

        lines = [
            line
            for line in lines
            if not line.strip().startswith("```")
        ]

        text = "\n".join(lines).strip()

        if text.lower().startswith("json"):

            text = text[4:].strip()

    # Find JSON object
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:

        raise ValueError(
            "No JSON object found in model response"
        )

    json_text = text[start:end + 1]

    return json.loads(json_text)


# ============================================================
# ANALYZE ONE IMAGE
# ============================================================

def analyze_image(
    image_path: Path,
    processor,
    model,
):

    print("\n" + "-" * 60)

    print(
        f"Processing: {image_path}"
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": build_prompt(
                        image_path.name
                    ),
                },
            ],
        }
    ]

    # --------------------------------------------------------
    # PREPARE INPUT
    # --------------------------------------------------------

    input_started = time.perf_counter()

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    for key, value in inputs.items():

        if isinstance(value, torch.Tensor):

            inputs[key] = value.to("cpu")

    input_time = (
        time.perf_counter()
        - input_started
    )

    # --------------------------------------------------------
    # GENERATION
    # --------------------------------------------------------

    print("Generating...")

    generation_started = time.perf_counter()

    with torch.inference_mode():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    generation_time = (
        time.perf_counter()
        - generation_started
    )

    # --------------------------------------------------------
    # REMOVE PROMPT TOKENS
    # --------------------------------------------------------

    prompt_length = (
        inputs["input_ids"].shape[1]
    )

    response_ids = (
        generated_ids[:, prompt_length:]
    )

    response = processor.batch_decode(
        response_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    print("\nRaw response:")
    print(response)

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    try:

        result = extract_json(response)

    except Exception as exc:

        print(
            f"[ERROR] JSON parsing failed: {exc}"
        )

        result = {
            "image": image_path.name,
            "jewellery_type": "unknown",
            "metal": "unknown",
            "metal_color": "unknown",
            "gemstones": [],
            "gemstone_count": 0,
            "design": "unknown",
            "chain_present": False,
            "approximate_objects": [],
            "error": str(exc),
            "raw_response": response,
        }

    # --------------------------------------------------------
    # SAVE JSON
    # --------------------------------------------------------

    result_path = image_path.with_name(
        image_path.stem + "_result.json"
    )

    result_path.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"\nInput preparation : {input_time:.2f} sec"
    )

    print(
        f"Generation time   : {generation_time:.2f} sec"
    )

    print(
        f"Saved JSON        : {result_path}"
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    total_started = time.perf_counter()

    images = find_images()

    if not images:

        print(
            "\nNo images found in:"
        )

        print(
            OUTPUT_DIR
        )

        return

    print(
        f"\nFound {len(images)} images."
    )

    print(
        "\nImages will be processed from:"
    )

    for folder in MODEL_OUTPUTS:

        print(
            f"  {OUTPUT_DIR / folder}"
        )

    processor, model = load_model()

    successful = 0
    failed = 0

    for index, image_path in enumerate(
        images,
        start=1,
    ):

        print(
            f"\n[{index}/{len(images)}]"
        )

        try:

            analyze_image(
                image_path,
                processor,
                model,
            )

            successful += 1

        except Exception as exc:

            failed += 1

            print(
                f"[ERROR] Failed: {image_path}"
            )

            print(
                f"Reason: {type(exc).__name__}: {exc}"
            )

    total_time = (
        time.perf_counter()
        - total_started
    )

    print("\n")
    print("=" * 60)
    print("QWEN3.5-2B ANALYSIS COMPLETE")
    print("=" * 60)

    print(
        f"Total images : {len(images)}"
    )

    print(
        f"Successful   : {successful}"
    )

    print(
        f"Failed       : {failed}"
    )

    print(
        f"Total time   : {total_time:.2f} sec"
    )

    print(
        f"Output       : {OUTPUT_DIR}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()