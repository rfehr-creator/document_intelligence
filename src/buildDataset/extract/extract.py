import json
from typing import List

from buildDataset.extract.rbcSaving import extractRBCSavingDirectory
from classes.FileMapping import FileMapping
from extract.mbHydro import extractMBHydroDirectory
from extract.rbcBank import extractRBCBankDirectory
from extract.rbcCredit import extractRBCCreditDirectory
from buildDataset.general import convert_pdf_page_to_image
import os

BASE_DEST_DIRECTORY = "../../data/dataset"


class Extract:
    def __init__(self):
        self.metadata = {"train": [], "validation": []}
        self.pdfToImg: List[FileMapping] = []

    def extract(self, memorize: bool = False):
        if memorize:
            # extract rbc credit card statement
            # rbcCredit = "../data/original/rbcCreditCardStatementsMem"
            # extractRBCCreditDirectory(rbcCredit, self.metadata, self.pdfToImg)
            rbcBank = "../../data/original/rbcBankStatementsMem"
            extractRBCBankDirectory(rbcBank, self.metadata, self.pdfToImg, BASE_DEST_DIRECTORY)
            return

        # extract rbc bank statement
        rbcBank = "../../data/original/rbcBankStatements"
        extractRBCBankDirectory(rbcBank, self.metadata, self.pdfToImg, BASE_DEST_DIRECTORY)

        # extract rbc saving statement
        rbcSaving = "../../data/original/rbcSavings"
        extractRBCSavingDirectory(rbcSaving, self.metadata, self.pdfToImg, BASE_DEST_DIRECTORY)

        # extract rbc credit card statement
        rbcCredit = "../../data/original/rbcCreditCardStatements"
        extractRBCCreditDirectory(rbcCredit, self.metadata, self.pdfToImg, BASE_DEST_DIRECTORY)

        # extract mb hydro statement
        mbHydro = "../../data/original/hydro"
        extractMBHydroDirectory(mbHydro, self.metadata, self.pdfToImg, BASE_DEST_DIRECTORY)

        # extract bell MTS statement
        bellMTS = "../../data/original/bellMTS"
        # extractBellMTSDirectory(bellMTS, self.metadata, self.pdfToImg, BASE_DEST_DIRECTORY)

    def writeMetadata(self):
        assert self.metadata, "Metadata is empty"
        # Write fileInfo to metadata.jsonl file in JSONL format
        with open(f"{BASE_DEST_DIRECTORY}/train/metadata.jsonl", "w") as trainMeta:
            with open(f"{BASE_DEST_DIRECTORY}/validation/metadata.jsonl", "w") as valMeta:
                for item in self.metadata["train"]:
                    trainMeta.write(json.dumps(item) + "\n")
                for item in self.metadata["validation"]:
                    valMeta.write(json.dumps(item) + "\n")

    def remove_files_in_directory(self, directory):
        file_list = os.listdir(directory)
        for file_name in file_list:
            file_path = os.path.join(directory, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                self.remove_files_in_directory(file_path)
                os.rmdir(file_path)

    def convertPDFToImages(self):
        assert self.pdfToImg, "PDF to Image mapping is empty"
        self.remove_files_in_directory(f"{BASE_DEST_DIRECTORY}/train/images")
        self.remove_files_in_directory(f"{BASE_DEST_DIRECTORY}/validation/images")
        for page in self.pdfToImg:
            assert os.path.exists(page.src), f"Src does not exist: {page.src}"
            convert_pdf_page_to_image(page.src, page.pageNumber, page.dest)
            assert os.path.exists(page.dest), f"Dest does not exist: {page.dest}"

    def run(self, memorize: bool = False):
        self.extract(memorize=memorize)
        self.writeMetadata()
        self.convertPDFToImages()
