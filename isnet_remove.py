import os
import sys
import time
import psutil
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DIS_ROOT = os.path.join(
    PROJECT_ROOT,
    "DIS"
)

INPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "input"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "output",
    "IS-Net"
)

ISNET_ROOT = os.path.join(
    DIS_ROOT,
    "IS-Net"
)

MODEL_PATH = os.path.join(
    DIS_ROOT,
    "saved_models",
    "IS-Net",
    "isnet-general-use.pth"
)

INPUT_SIZE = (1024, 1024)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# IMPORT OFFICIAL IS-NET
# ============================================================

sys.path.insert(0, ISNET_ROOT)

from models.isnet import ISNetDIS


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("IS-Net Jewellery Background Removal Benchmark")
print("=" * 60)

print(f"Device: {DEVICE}")
print(f"Model: {MODEL_PATH}")
print(f"Input: {INPUT_DIR}")
print(f"Output: {OUTPUT_DIR}")


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"\nIS-Net checkpoint not found:\n{MODEL_PATH}\n"
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading IS-Net...")

model = ISNetDIS()

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(checkpoint)

model = model.to(DEVICE)
model.eval()

print("IS-Net loaded successfully!")


# ============================================================
# IMAGE LIST
# ============================================================

extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
)

image_files = sorted(
    [
        f
        for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(extensions)
    ]
)

if not image_files:
    raise RuntimeError(
        f"No images found in {INPUT_DIR}"
    )

print(f"\nFound {len(image_files)} images.")


# ============================================================
# PROCESSOR MEMORY
# ============================================================

process = psutil.Process(os.getpid())


# ============================================================
# RESULTS
# ============================================================

times = []
ram_changes = []
ram_usages = []


# ============================================================
# PROCESS ALL IMAGES
# ============================================================

for index, filename in enumerate(
    image_files,
    start=1
):

    input_path = os.path.join(
        INPUT_DIR,
        filename
    )

    output_name = (
        os.path.splitext(filename)[0]
        + ".png"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        output_name
    )

    print("\n" + "-" * 60)
    print(
        f"[{index}/{len(image_files)}] {filename}"
    )

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    image = Image.open(
        input_path
    ).convert("RGB")

    original_width, original_height = image.size

    image_np = np.array(image)

    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    image_tensor = torch.from_numpy(
        image_np
    ).float()

    image_tensor = image_tensor.permute(
        2,
        0,
        1
    ).unsqueeze(0)

    image_tensor = F.interpolate(
        image_tensor,
        size=INPUT_SIZE,
        mode="bilinear",
        align_corners=False
    )

    image_tensor = image_tensor.to(
        dtype=torch.uint8
    )

    image_tensor = (
        image_tensor.float() / 255.0
    )

    # Official IS-Net normalization
    image_tensor = (
        image_tensor - 0.5
    ) / 1.0

    image_tensor = image_tensor.to(
        DEVICE
    )

    # --------------------------------------------------------
    # MEMORY BEFORE INFERENCE
    # --------------------------------------------------------

    ram_before = (
        process.memory_info().rss
        / (1024 * 1024)
    )

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.no_grad():

        result = model(
            image_tensor
        )

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    end_time = time.perf_counter()

    inference_time = (
        end_time - start_time
    )

    # --------------------------------------------------------
    # MEMORY AFTER INFERENCE
    # --------------------------------------------------------

    ram_after = (
        process.memory_info().rss
        / (1024 * 1024)
    )

    ram_change = (
        ram_after - ram_before
    )

    # --------------------------------------------------------
    # EXTRACT MASK
    # --------------------------------------------------------

    # IS-Net returns multiple outputs.
    # The first output contains the segmentation prediction.

    if isinstance(
        result,
        (list, tuple)
    ):
        prediction = result[0]
    else:
        prediction = result

    # Some IS-Net versions return nested outputs.
    if isinstance(
        prediction,
        (list, tuple)
    ):
        prediction = prediction[0]

    prediction = prediction[
        :,
        0:1,
        :,
        :
    ]

    # --------------------------------------------------------
    # RESIZE MASK TO ORIGINAL IMAGE SIZE
    # --------------------------------------------------------

    prediction = F.interpolate(
        prediction,
        size=(
            original_height,
            original_width
        ),
        mode="bilinear",
        align_corners=False
    )

    prediction = prediction.squeeze()

    prediction = prediction.cpu().numpy()

    # --------------------------------------------------------
    # NORMALIZE MASK
    # --------------------------------------------------------

    mask_min = prediction.min()
    mask_max = prediction.max()

    if mask_max > mask_min:

        prediction = (
            prediction - mask_min
        ) / (
            mask_max - mask_min
        )

    else:

        prediction = np.zeros_like(
            prediction
        )

    mask = (
        prediction * 255
    ).clip(
        0,
        255
    ).astype(
        np.uint8
    )

    # --------------------------------------------------------
    # CREATE TRANSPARENT PNG
    # --------------------------------------------------------

    rgba = image.convert(
        "RGBA"
    )

    alpha = Image.fromarray(
        mask,
        mode="L"
    )

    rgba.putalpha(
        alpha
    )

    rgba.save(
        output_path,
        "PNG"
    )

    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    times.append(
        inference_time
    )

    ram_changes.append(
        ram_change
    )

    ram_usages.append(
        ram_after
    )

    # --------------------------------------------------------
    # PRINT RESULT
    # --------------------------------------------------------

    print(
        f"Inference time : "
        f"{inference_time:.3f} sec"
    )

    print(
        f"RAM before     : "
        f"{ram_before:.2f} MB"
    )

    print(
        f"RAM after      : "
        f"{ram_after:.2f} MB"
    )

    print(
        f"RAM change     : "
        f"{ram_change:+.2f} MB"
    )

    print(
        f"Saved          : "
        f"{output_path}"
    )


# ============================================================
# FINAL STATISTICS
# ============================================================

print("\n")
print("=" * 60)
print("IS-Net BENCHMARK RESULTS")
print("=" * 60)

total_time = sum(times)

average_time = np.mean(times)
minimum_time = np.min(times)
maximum_time = np.max(times)

average_ram_change = np.mean(
    ram_changes
)

average_ram_usage = np.mean(
    ram_usages
)

maximum_ram_usage = np.max(
    ram_usages
)

images_per_hour = (
    3600 / average_time
    if average_time > 0
    else 0
)


print(
    f"Images processed       : "
    f"{len(image_files)}"
)

print(
    f"Total inference time   : "
    f"{total_time:.3f} sec"
)

print(
    f"Average inference time : "
    f"{average_time:.3f} sec/image"
)

print(
    f"Minimum inference time : "
    f"{minimum_time:.3f} sec"
)

print(
    f"Maximum inference time : "
    f"{maximum_time:.3f} sec"
)

print(
    f"Average RAM change     : "
    f"{average_ram_change:+.2f} MB"
)

print(
    f"Average process RAM    : "
    f"{average_ram_usage:.2f} MB"
)

print(
    f"Maximum process RAM    : "
    f"{maximum_ram_usage:.2f} MB"
)

print(
    f"Estimated images/hour  : "
    f"{images_per_hour:.1f}"
)

print("\nOutput folder:")

print(OUTPUT_DIR)

print(
    "\nIS-Net processing completed successfully!"
)