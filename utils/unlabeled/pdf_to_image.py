"""
    Convert PDFs to images. 

    Usage:
        Config the DATA_ROOT, OCR, PDF, ANNOT, SAVE_ROOT, CONVERTED variables.

        Run the script with
        python pdf_to_image.py

    Note:
        This script is not used in the training and inferencing pipeline. It is only used to convert PDFs to images.
        The exported images can be used as cached images for Training and Inferencing.
"""


import pathlib
import json
import os
import numpy as np
import tqdm
import matplotlib.pyplot as plt

from pdf2image import convert_from_path


DATA_ROOT = pathlib.Path("/docile/unlabeled/chunk-01/original")
OCR = DATA_ROOT / "ocr"
PDF = DATA_ROOT / "pdfs"
ANNOT = DATA_ROOT / "annotations"
SAVE_ROOT = "converted_images"
CONVERTED = os.listdir("converted_images")


if __name__ == "__main__":
    pdfs = list(PDF.iterdir())

    for pdf_path in tqdm.tqdm(pdfs):
        fname = str(pdf_path).split("/")[-1][:-4]

        if fname + ".png" in CONVERTED:
            continue

        annotations = json.load(open(f"{ANNOT}/{fname}.json"))

        # filter out images that are not in the cluster range
        if not (
            annotations["metadata"]["cluster_id"] >= 0
            and annotations["metadata"]["cluster_id"] <= 1070
        ):
            continue

        # filter out images that are too big
        if len(annotations["metadata"]["page_sizes_at_200dpi"]) > 0:
            t = annotations["metadata"]["page_sizes_at_200dpi"][0]

            if t[0] >= 3000 and t[1] >= 3000:
                continue

        ocr = json.load(open(f"{OCR}/{fname}.json"))
        cnt = 0
        cnt_below_thres = 0

        for block in ocr["pages"][0]["blocks"]:
            for line in block["lines"]:
                for word in line["words"]:
                    conf = word["confidence"]

                    if conf < 0.5:
                        cnt_below_thres += 1

                    cnt += 1

        # if count ocr of 1 page is below 20
        # or if more than half of the ocr is below 0.5 confidence
        if cnt_below_thres > cnt / 2 or cnt < 20:
            continue

        images = convert_from_path(pdf_path, thread_count=16)

        if not os.path.exists(SAVE_ROOT):
            os.mkdir(SAVE_ROOT)

        if not os.path.exists(f"{SAVE_ROOT}/" + fname):
            os.makedirs(f"{SAVE_ROOT}/" + fname)

        for index, image in enumerate(images):
            image.save(f"{SAVE_ROOT}/{fname}/{index}.png")