from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
import json
import os
import random
import re
from typing import Callable, List, Tuple
from pdf2image import convert_from_path
import PyPDF2
from classes.FileMapping import FileMapping
from classes.JsonKey import Key

# Load configuration from JSON file
with open("config.json", "r") as f:
    config = json.load(f)

datasetSplit = config["dataset"]["split"]
trainSplit = datasetSplit["train"]
valSplit = datasetSplit["val"]

def readPdfText(filename):
    text_list = []
    with open(filename, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        
         # Check if the PDF is encrypted
        if reader.is_encrypted:
            reader.decrypt("")  # Attempt to decrypt with an empty password
            
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text_list.append(page.extract_text())
    return text_list


def shortMonthToNumber(month: str) -> int:
    return {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }[month.lower()[:3]]


def longMonthToNumber(month: str) -> int:
    return shortMonthToNumber(month)


def convert_pdf_page_to_image(
    pdf_filename: str, page_number: int, output_image_path: str
):
    # Convert the specified page to an image
    images = convert_from_path(
        pdf_filename, first_page=page_number, last_page=page_number
    )

    # Save the image to the specified location
    if images:
        images[0].save(output_image_path)


def extractAccountNumber(text: str, pattern: str) -> str | None:
    accountNumber = re.search(pattern, text)
    if accountNumber and accountNumber.group(1):
        return accountNumber.group(1).replace(" ", "")
    return None


def extractDate(text: str, pattern: str, year: int = 0) -> str | None:
    date = re.search(pattern, text)
    groups = date.groups()
    print(groups)
    if len(groups) == 3:
        return {
            Key.day: int(date.group(2)),
            Key.month: shortMonthToNumber(date.group(1)),
            Key.year: int(date.group(3)),
        }
    elif len(groups) == 2:
        return {
            Key.day: int(date.group(2)),
            Key.month: shortMonthToNumber(date.group(1)),
            Key.year: year,
        }
    return None


def toDecimal(amount: str) -> Decimal:
    return Decimal(amount.replace(",", "")).quantize(
        Decimal("0.00"), rounding=ROUND_HALF_UP
    )


def extractMoney(text: str, pattern: str) -> Decimal | None:
    money = re.search(pattern, text)
    if money and money.group(1):
        return toDecimal(money.group(1))
    return None


def extractMinusMoney(
    text: str, pattern: str, minusSign: str, amountGroup=2, minusGroup=1
) -> Decimal | None:
    money = re.search(pattern, text)

    if money and len(money.groups()) == 2:
        if money.group(minusGroup) == minusSign:
            return -1 * toDecimal(money.group(amountGroup))
        return toDecimal(money.group(amountGroup))

    return None


def extractStartEndDate(
    text: str, generalPattern: str, shortPattern: str
) -> Tuple[str, str] | Tuple[None, None]:
    dates = re.search(generalPattern, text)
    if dates and len(dates.groups()) == 2:
        startDate = extractDate(dates.group(1), shortPattern)
        endDate = extractDate(dates.group(2), shortPattern)
        if startDate and endDate:
            return (startDate, endDate)
    return (None, None)


def processFile(
    filepath: str,
    vendorType: str,
    baseDestDirectory: str,
    fileInfoProcessor: Callable[[str], List[str]],
    printText=False,
) -> Tuple[List[str], List[FileMapping]]:
    assert os.path.exists(filepath), "File does not exist"
    assert str(filepath).endswith(".pdf"), "File is not a PDF"
    assert callable(fileInfoProcessor), "fileInfoProcessor is not a function"
    assert vendorType, "vendorType is empty"

    print(f"Processing File: {filepath}")

    pageInfo = {"train": [], "validation": []}
    fileInfo: List[FileMapping] = []

    text = readPdfText(filepath)
    if printText:
        print(text)

    info = fileInfoProcessor(text)
    assert len(text) == len(
        info
    ), f"Length of text and info should be the same for : {filepath}"

    for index, i in enumerate(info):
        image_path = f"images/{vendorType}_{i[Key.pageNumber]}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.jpg"

        # Add the file mapping to the list
        dataset_type = random.choices(["train", "validation"], weights=[trainSplit, valSplit], k=1)[0]
        fileInfo.append(
            FileMapping(filepath, f"{baseDestDirectory}/{dataset_type}/{image_path}", index + 1)
        )

        # Add the target text to the list
        pageInfo[dataset_type].append(
            {"file_name": image_path, "ground_truth": json.dumps({"gt_parse": i})}
        )
    assert len(pageInfo["train"]) + len(pageInfo["validation"]) > 0, "pageInfo train and validation are empty in processFile"
    assert len(fileInfo), "fileInfo is empty in processFile"
    return (pageInfo, fileInfo)


def extractPageNumber(text: str, pattern: str, groupIdx=1) -> str | None:
    pageNumber = re.search(pattern, text)
    if pageNumber and pageNumber.group(groupIdx):
        return int(pageNumber.group(groupIdx))

    return None


def iterateDirectory(
    path: str,
    pageInfo: dict,
    fileInfo: List[FileMapping],
    file_processor: Callable[[str], Tuple[List[str], List[FileMapping]]],
    baseDestDirectory: str,
):
    assert os.path.exists(path), "Directory does not exist"
    assert callable(file_processor), "file_processor is not a function"

    for dirPath, _, filenames in os.walk(path):
        print(dirPath)
        for file in filenames:
            print(file)
            newPageInfo, newFileInfo = file_processor(f"{path}/{file}", baseDestDirectory)

            assert isinstance(newPageInfo, dict), "newPageInfo is not a list"
            assert len(newPageInfo["train"]) + len(newPageInfo["validation"]) > 0, "newPageInfo train and validation are empty in iterateDirectory"
            
            pageInfo["train"].extend(newPageInfo["train"])
            pageInfo["validation"].extend(newPageInfo["validation"])

            assert isinstance(newFileInfo, list), "newFileInfo is not a list"
            assert len(newFileInfo), "newFileInfo is empty"
            fileInfo.extend(newFileInfo)


def compareFloats(a: float, b: float, epsilon: float = 0.01) -> bool:
    return abs(a - b) < epsilon
