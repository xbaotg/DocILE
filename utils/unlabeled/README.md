In order to use unlabeled data for training, you need to use predictions from models. In this case, I used the output when I inference three models with `Ensemble`.:

1. You need to download `unlabeled data` and their `annotations`

2. Make them have a structure as same to this

├── annotations        
├── ocr                 
└── pdfs

Each of them must contain corresponding annotations/ocr/pdfs to document in `downloaded unlabeled data`.

3. Config path in `pdf_to_image.py`, execute that script to preprocess unlabeled data (filter out the rotated documents, images have a size too big, ...), and convert documents to images, which can be used as `cached` when training or inference. (When training or inference, we don't need to convert them again).

4. Use models to make predictions (in this case, Inference with ensemble method) on unlabeled data, remember to config the path to the cached folder which is created in step 3, it will increase inference speed much.

5. Config path in `create_pseudo_data.py` to the `KILE_predictions` and `LIR_predictions` which are the output of step 4. After that execute the script, it will create pseudo data which contains `unlabeled` and `validation` documents.

6. Train models with the pseudo data.
