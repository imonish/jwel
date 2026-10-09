import os
import time
import psutil
import torch

from PIL import Image
from transformers import AutoProcessor, AutoModelForImageSegmentation


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "ZhengPeng7/BiRefNet"

INPUT_DIR = "data/input"
OUTPUT_DIR = "data/output/BiRefNet"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

process = psutil.Process(os.getpid())

print("=" * 70)
print("BiRefNet Jewellery Background Removal Benchmark")
print("=" * 70)

print("Device:", device)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading BiRefNet...")

processor = AutoProcessor.from_pretrained(MODEL_NAME)

model = AutoModelForImageSegmentation.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = model.to(device)
model.eval()

print("BiRefNet loaded successfully!")


# ============================================================
# FIND IMAGES
# ============================================================

extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
)

image_files = sorted([
    f
    for f in os.listdir(INPUT_DIR)
    if f.lower().endswith(extensions)
])


print("\nImages found:", len(image_files))

if not image_files:
    print("No images found in:", INPUT_DIR)
    exit()


# ============================================================
# BENCHMARK STORAGE
# ============================================================

times = []
ram_usages = []


# ============================================================
# PROCESS IMAGES
# ============================================================

for index, filename in enumerate(image_files, start=1):

    input_path = os.path.join(
        INPUT_DIR,
        filename
    )

    base_name = os.path.splitext(filename)[0]

    output_path = os.path.join(
        OUTPUT_DIR,
        base_name + ".png"
    )

    print("\n" + "-" * 70)
    print(f"[{index}/{len(image_files)}] {filename}")

    try:

        # ----------------------------------------------------
        # LOAD IMAGE
        # ----------------------------------------------------

        image = Image.open(input_path).convert("RGB")

        # ----------------------------------------------------
        # RAM BEFORE
        # ----------------------------------------------------

        ram_before = process.memory_info().rss / (1024 ** 2)

        # ----------------------------------------------------
        # START TIMER
        # ----------------------------------------------------

        start_time = time.perf_counter()

        # ----------------------------------------------------
        # PREPROCESS
        # ----------------------------------------------------

        inputs = processor(
            images=image,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        # ----------------------------------------------------
        # INFERENCE
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = model(**inputs)

        # ----------------------------------------------------
        # GET PREDICTION
        # ----------------------------------------------------

        prediction = outputs.logits

        prediction = torch.sigmoid(prediction)

        prediction = prediction.squeeze()

        # ----------------------------------------------------
        # CONVERT MASK TO IMAGE
        # ----------------------------------------------------

        mask = prediction.cpu()

        mask = (mask * 255).clamp(0, 255).byte()

        mask_image = Image.fromarray(
            mask.numpy(),
            mode="L"
        )

        # Resize mask to original image size
        mask_image = mask_image.resize(
            image.size,
            Image.Resampling.LANCZOS
        )

        # ----------------------------------------------------
        # CREATE RGBA IMAGE
        # ----------------------------------------------------

        output = image.convert("RGBA")

        output.putalpha(mask_image)

        # ----------------------------------------------------
        # STOP TIMER
        # ----------------------------------------------------

        end_time = time.perf_counter()

        inference_time = end_time - start_time

        # ----------------------------------------------------
        # RAM AFTER
        # ----------------------------------------------------

        ram_after = process.memory_info().rss / (1024 ** 2)

        ram_used = ram_after - ram_before

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        output.save(
            output_path,
            "PNG"
        )

        # ----------------------------------------------------
        # STORE RESULTS
        # ----------------------------------------------------

        times.append(inference_time)
        ram_usages.append(ram_used)

        print(
            f"Inference time : {inference_time:.3f} sec"
        )

        print(
            f"RAM change     : {ram_used:.2f} MB"
        )

        print(
            f"Process RAM    : {ram_after:.2f} MB"
        )

        print(
            f"Saved          : {output_path}"
        )

    except Exception as e:

        print("\nERROR:", filename)
        print(e)


# ============================================================
# FINAL BENCHMARK
# ============================================================

print("\n")
print("=" * 70)
print("BiRefNet BENCHMARK RESULTS")
print("=" * 70)

if times:

    avg_time = sum(times) / len(times)

    min_time = min(times)

    max_time = max(times)

    total_time = sum(times)

    avg_ram = sum(ram_usages) / len(ram_usages)

    print(
        f"Images processed       : {len(times)}"
    )

    print(
        f"Total inference time   : {total_time:.3f} sec"
    )

    print(
        f"Average time / image   : {avg_time:.3f} sec"
    )

    print(
        f"Minimum time / image   : {min_time:.3f} sec"
    )

    print(
        f"Maximum time / image   : {max_time:.3f} sec"
    )

    print(
        f"Average RAM change     : {avg_ram:.2f} MB"
    )

    print(
        f"Estimated images/hour  : {3600 / avg_time:.1f}"
    )

else:

    print("No images were processed successfully.")

print("=" * 70)