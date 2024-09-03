# Document Intelligence

This project focuses on extracting information from downloaded PDFs to create training data for the Donut Model on Huggingface and then finetune it. After extraction, the PDFs are converted to images with target data. Since building the dataset will be unique for everyone, this project is meant to be a starting point where everyone can modify it to their specific needs. 

## Problem

The original Donut model exhibited inconsistent performance after saving and reloading. To address this issue, I am modifying the configuration then saving the model and processor and then reload them and then finetune.

## Todo

Due to time constraints this summer, at least the following improvements are needed for this repository:

- Better organization of the code
- Consistent naming conventions
- Batch size greater than 1 is not working

## Build Dataset
- Run `python3 run.py` in buildDataset folder

## Train
- Run `python3 run.py` in train folder

## Inference on Single Image
- Run `python3 inference.py` in inference folder