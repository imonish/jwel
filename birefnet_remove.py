import os
import time
import psutil
import torch

from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "ZhengPeng7/BiRefNet_lite"

INPUT_DIR = "data/input"
OUTPUT_DIR = "data/output"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

process = psutil.Process(os.getpid())

print("=" * 70)
print("BiRefNet-Lite Benchmark")
print("=" * 70)
print("Device:", device)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading BiRefNet-Lite...")

model = AutoModelForImageSegmentation.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = model.to(device)
model.eval()

print("Model loaded successfully!")


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


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
    print("No images found.")
    exit()


# ============================================================
# BENCHMARK STORAGE
# ============================================================

times = []
ram_usages = []


# ============================================================
# PROCESS
# ============================================================

for index, filename in enumerate(image_files, start=1):

    input_path = os.path.join(INPUT_DIR, filename)

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

        original_size = image.size

        # ----------------------------------------------------
        # PREPROCESS
        # ----------------------------------------------------

        input_tensor = transform(image).unsqueeze(0).to(device)

        # ----------------------------------------------------
        # RAM BEFORE
        # ----------------------------------------------------

        ram_before = process.memory_info().rss / (1024 ** 2)

        # ----------------------------------------------------
        # START TIMER
        # ----------------------------------------------------

        start_time = time.perf_counter()

        # ----------------------------------------------------
        # INFERENCE
        # ----------------------------------------------------

        with torch.no_grad():

            prediction = model(input_tensor)[-1].sigmoid()

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
        # CREATE MASK
        # ----------------------------------------------------

        mask = prediction[0].squeeze().cpu()

        mask = transforms.ToPILImage()(mask)

        mask = mask.resize(
            original_size,
            Image.Resampling.LANCZOS
        )

        # ----------------------------------------------------
        # CREATE TRANSPARENT PNG
        # ----------------------------------------------------

        output_image = image.convert("RGBA")

        output_image.putalpha(mask)

        output_image.save(
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
            f"RAM process    : {ram_after:.2f} MB"
        )

        print(
            f"Saved          : {output_path}"
        )

    except Exception as e:

        print("ERROR:", filename)
        print(e)


# ============================================================
# FINAL BENCHMARK
# ============================================================

print("\n")
print("=" * 70)
print("BENCHMARK RESULTS")
print("=" * 70)

if times:

    avg_time = sum(times) / len(times)

    min_time = min(times)

    max_time = max(times)

    total_time = sum(times)

    avg_ram = sum(ram_usages) / len(ram_usages)

    print(f"Images processed       : {len(times)}")

    print(
        f"Total inference time  : {total_time:.3f} sec"
    )

    print(
        f"Average time / image  : {avg_time:.3f} sec"
    )

    print(
        f"Minimum time / image  : {min_time:.3f} sec"
    )

    print(
        f"Maximum time / image  : {max_time:.3f} sec"
    )

    print(
        f"Average RAM change    : {avg_ram:.2f} MB"
    )

    print(
        f"Estimated images/hour : {3600 / avg_time:.1f}"
    )

print("=" * 70)