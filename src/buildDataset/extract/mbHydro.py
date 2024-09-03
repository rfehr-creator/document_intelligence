from typing import List, Tuple
from classes.FileMapping import FileMapping
from classes.JsonKey import Key

from buildDataset.general import (
    extractAccountNumber,
    extractDate,
    extractMinusMoney,
    extractMoney,
    extractPageNumber,
    iterateDirectory,
    processFile,
)


def extractMBHydroInfo(text: List[str]) -> List[str]:
    rbcBankInfo = []
    for index, pageText in enumerate(text):
        json = {}

        pageNumber = (
            1
            if index == 0
            else extractPageNumber(pageText, r"(\d+) / (\d+)", groupIdx=1)
        )
        json[Key.pageNumber] = pageNumber
        assert json[Key.pageNumber] >= 1, "Page number not found for MB Hydro"

        if pageNumber == 1:
            prevBalance = extractMinusMoney(
                pageText, r"Balance\s+forward\D+([\d,]+\.\d{2})\s?(CR)?", "CR", 1, 2
            )
            electricity = extractMoney(
                pageText, r"Electricity\D+[\d,]+\.\d{2}\D+([\d,]+\.\d{2})"
            )
            eppDiff = (
                extractMinusMoney(
                    pageText,
                    r"EPP\s+Diff\D+[\d,]+\.\d{2}\D+([\d,]+\.\d{2})\s?(CR)?",
                    "CR",
                    1,
                    2,
                )
                or 0
            )
            newBalance = extractMinusMoney(
                pageText, r"Amount due\D+([\d,]+\.\d{2})\s?(CR)?", "CR", 1, 2
            )

            print(prevBalance, electricity, eppDiff, newBalance)
            assert (
                prevBalance + electricity + eppDiff == newBalance
            ), "Amounts do not match for MB Hydro"

            if eppDiff > 0:
                json[Key.eppDiff] = str(eppDiff)

            json[Key.newBalance] = str(newBalance)
            json[Key.newCharges] = str(electricity)
            json[Key.prevBalance] = str(prevBalance)

        if "Consumption history" not in pageText:
            accountNumber = extractAccountNumber(pageText, r"(\d{7}\D{1}\d{7})")
            billDate = extractDate(pageText, r"mission(\w{3})\W(\d{1,2})\D{5}(\d{4})")

            json[Key.accountNumber] = accountNumber
            assert accountNumber, "Account number not found for MB Hydro"

            json[Key.billDate] = billDate
            assert billDate, "Bill date not found for Mb Hydro"

            print("-----------------------------")
            print("Index:", index)
            print("Page Number:", pageNumber)
            print("Bill Date:", billDate)
            print("Account Number:", accountNumber)
            print("Previous Balance:", prevBalance)
            print("Electricity:", electricity)
            print("New Balance:", newBalance)
            print("-----------------------------")
        else:
            print("-----------------------------")
            print("Index:", index)
            print("Page Number:", pageNumber)
            print("-----------------------------")

        json[Key.type] = "Bill"
        json[Key.vendor] = "MB Hydro"

        rbcBankInfo.append(json)

    return rbcBankInfo


def extractMBHydroFile(filepath: str, baseDestDirectory: str) -> Tuple[List[str], List[FileMapping]]:
    return processFile(filepath, "mb_hydro", baseDestDirectory, extractMBHydroInfo)


def extractMBHydroDirectory(
    directoryPath: str, pageInfo: List[str], fileInfo: List[FileMapping], baseDestDirectory: str
):
    iterateDirectory(directoryPath, pageInfo, fileInfo, extractMBHydroFile, baseDestDirectory)
