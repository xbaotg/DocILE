import cv2
import sys
import numpy as np
import json

from docile.dataset import Dataset
from docile.dataset.bbox import BBox
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, "../")
from helpers import FieldWithGroups


def visulize_ocrs(ocr, img, W, H):
    for field in ocr:
        bbox = field.bbox.to_absolute_coords(W, H)
        text = field.text

        img = cv2.rectangle(img, (bbox.left, bbox.top), (bbox.right, bbox.bottom), (255, 0, 0), 1)
        img = cv2.putText(img, text, (bbox.left, bbox.top), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)


def visualize_predictions(predictions, docid, page_idx, img, W, H):
    predicted = predictions[docid]

    for pred in predicted:
        if pred['page'] == page_idx:
            bbox = BBox(*pred['bbox']).to_absolute_coords(W, H)
            fieldtype = pred["fieldtype"]

            img = cv2.rectangle(img, (bbox.left, bbox.top), (bbox.right, bbox.bottom), (0, 0, 255), 1)
            img = cv2.putText(img, fieldtype, (bbox.left, bbox.top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)


def visualize_annotations(kile_fields_page, img):
    for field in kile_fields_page:
        bbox = field.bbox
        fieldtype = field.fieldtype

        img = cv2.rectangle(img, (bbox.left, bbox.top), (bbox.right, bbox.bottom), (0, 0, 255), 1)
        img = cv2.putText(img, fieldtype, (bbox.left, bbox.top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)


def main():
    # Change this to your own path
    data = Dataset("test", "/docile/data/testset", load_annotations=False, load_ocr=True)

    # Change this to your own path, "predictions.json" is the output when you inference
    predictions = json.load(open("/docile/scripts/upper_bound/submit.json"))

    if not Path("output").exists():
        Path("output").mkdir()

    USE_ANNOTATIONS = False
    USE_OCR = True
    USE_PREDICTIONS = False

    for document in tqdm(data):
        docid = document.docid
        page_count = document.page_count

        if USE_ANNOTATIONS:
            kile_fields = [
                FieldWithGroups.from_dict(field.to_dict()) for field in document.annotation.fields
            ]

        for page_idx in range(page_count):
            img = document.page_image(page_idx)
            W, H = img.size

            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            if USE_OCR:
                ocr = document.ocr.get_all_words(page_idx, snapped=False, use_cached_snapping=True, get_page_image=lambda: img)
                visulize_ocrs(ocr, img, W, H)

            if USE_PREDICTIONS:
                visualize_predictions(predictions, docid, page_idx, img, W, H)

            if USE_ANNOTATIONS:
                kile_fields_page = [field for field in kile_fields if field.page == page_idx]
                visualize_annotations(kile_fields_page, img)

            cv2.imwrite(f"output/{docid}_{page_idx}.png", img)


if __name__ == "__main__":
    main()