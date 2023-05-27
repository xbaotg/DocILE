"""
    Using the predictions from KILE and LIR predictions which is the output of inference, 
    after that, we will create pseudo data for training
"""

import json
import os
import shutil

from pathlib import Path
from tqdm import tqdm


KILE = json.load(
    open(
        "/docile/predictions/roberta_base_with_synthetic_pretraining/unlabeled_predictions_KILE.json"
    )
)
LIR = json.load(
    open(
        "/docile/predictions/roberta_base_with_synthetic_pretraining/unlabeled_predictions_LIR.json"
    )
)
SAVE_TO = "pseudo_data"

# list of fname we need to create pseudo data
FILES = os.listdir("converted_images")

# path to unlabeled data
UNLABELED_PATH = "/docile/unlabeled/chunk-01/original/"

# path to annotated data
ANNOTATED_PATH = "/mlcv/WorkingSpace/Personals/thinhvq/DocILE/docile/data_train"

# create folder if not exists
FOLDER_CREATE = ["", "annotations", "ocr", "pdfs"]
for folder in FOLDER_CREATE:
    if not Path(f"{SAVE_TO}/{folder}").exists():
        Path(f"{SAVE_TO}/{folder}").mkdir()


FNAMES = []

for fname in tqdm(FILES):
    if fname not in KILE and fname not in LIR:
        continue

    FNAMES.append(fname)
    meta_data = json.load(open(f"{UNLABELED_PATH}/annotations/{fname}.json"))

    output = {
        "field_extractions": [],
        "line_item_extractions": [],
        "metadata": meta_data["metadata"],
    }

    output["metadata"]["currency"] = "usd"
    output["metadata"]["document_type"] = "order"
    output["metadata"]["language"] = "eng"
    output["metadata"]["page_to_table_grid"] = {}

    if fname in KILE:
        pred = KILE[fname]

        for each in pred:
            each.pop("line_item_id")
            each.pop("groups")
            score = float(each["score"])
            output["field_extractions"].append(each)

    if fname in LIR:
        pred = LIR[fname]

        for each in pred:
            each.pop("groups")
            score = float(each["score"])
            output["line_item_extractions"].append(each)

    with open(f"{SAVE_TO}/annotations/{fname}.json", "w+") as fout:
        fout.write(json.dumps(output))

# copy ocr, pdfs, from original -> train
with open(f"{SAVE_TO}/pseudo.json", "w+") as fout:
    fout.write(json.dumps(FNAMES))

for fname in os.listdir(f"{UNLABELED_PATH}/ocr"):
    if fname[:-5] not in FNAMES:
        continue

    shutil.copy(f"{UNLABELED_PATH}/ocr/{fname}", f"{SAVE_TO}/ocr/{fname}")

for fname in os.listdir(f"{UNLABELED_PATH}/pdfs"):
    if fname[:-4] not in FNAMES:
        continue

    shutil.copy(f"{UNLABELED_PATH}/pdfs/{fname}", f"{SAVE_TO}/pdfs/{fname}")

# # copy from val -> folder
shutil.copy(f"{ANNOTATED_PATH}/val.json", f"{SAVE_TO}/val.json")

for fname in json.load(open(f'{ANNOTATED_PATH}/val.json')):
    shutil.copy(f"{ANNOTATED_PATH}/annotations/{fname}.json", f"{SAVE_TO}/annotations/{fname}.json")
    shutil.copy(f"{ANNOTATED_PATH}/ocr/{fname}.json", f"{SAVE_TO}/ocr/{fname}.json")
    shutil.copy(f"{ANNOTATED_PATH}/pdfs/{fname}.pdf", f"{SAVE_TO}/pdfs/{fname}.pdf")