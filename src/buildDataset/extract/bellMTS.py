from typing import List, Tuple
from classes.FileMapping import FileMapping
from classes.JsonKey import Key

from buildDataset.general import (
    extractAccountNumber,
    extractDate,
    extractMinusMoney,
    extractPageNumber,
    iterateDirectory,
    processFile,
)
from classes.JsonTokenConstants import TokenConstants


def extractBellMTSInfo(text: List[str]) -> List[str]:
    infoList = []
    for index, pageText in enumerate(text):
        json = {}

        pageNumber = (
            1
            if index == 0
            else extractPageNumber(pageText, r"Page (\d+) Of (\d+)", groupIdx=1)
        )
        assert pageNumber >= 1, "Page number not found for Bell MTS"
        
        json[Key.pageNumber] = pageNumber

        if pageNumber == 1:
            accountNumber = extractAccountNumber(pageText, r"\D+(\d{9})\D+")
            
            billDate = extractDate(pageText, r"\wnvoice Date (\w+) (\d{2}), (\d{4})")
            newBalance = extractMinusMoney(
                pageText, r"Amount Due\D+([\d,]+\.\d{2})\s?(Credit)?", "Credit", 1, 2
            )
            
            assert accountNumber, "Account number not found for Bell MTS"
            assert newBalance, "New balance not found for Bell MTS"
            assert billDate, "Bill date not found for Bell MTS"
            
            json[Key.accountNumber] = accountNumber 
            json[Key.billDate] = billDate
            json[Key.newBalance] = str(newBalance)
            
            print("-----------------------------")
            print("Index:", index)
            print("Page Number:", pageNumber)
            print("Account Number:", accountNumber)
            print("Bill Date:", billDate)
            print("New Balance:", newBalance)
            print("-----------------------------")
        elif pageNumber == 2:
            prevBalance = extractMinusMoney(
                pageText, r"(?:Sub-Total|Outstanding Balance)\D+([\d,]+\.\d{2})\s?(Credit)?", "Credit", 1, 2
            )
            internet = extractMinusMoney(
                pageText, r"TOTAL INTERNET\D+([\d,]+\.\d{2})\s?(Credit)?", "Credit", 1, 2
            )
            gst = extractMinusMoney(
                pageText, r"GST \d\D+([\d,]+\.\d{2})\s?(Credit)?", "Credit", 1, 2
            )
            pst = extractMinusMoney(
                pageText, r"MB PST \d\D+([\d,]+\.\d{2})\s?(Credit)?", "Credit", 1, 2
            )
            
            print(prevBalance, internet, gst, pst, newBalance)
            assert prevBalance + internet + gst + pst == newBalance, "Amounts do not match for Bell MTS"
            
            json[Key.gst] = str(gst)
            json[Key.internet] = str(internet)
            json[Key.prevBalance] = str(prevBalance)
            json[Key.pst] = str(pst)
            
            print("-----------------------------")
            print("Index:", index)
            print("Page Number:", pageNumber)
            print("Account Number:", accountNumber)
            print("Bill Date:", billDate)
            print("Previous Balance:", prevBalance)
            print("Internet:", internet)
            print("GST:", gst)
            print("PST:", pst)
            print("-----------------------------")
        
        else:
            assert False, "Invalid page number for Bell MTS"


        json[Key.type] = TokenConstants.BILL
        json[Key.vendor] = TokenConstants.BELL_MTS

        infoList.append(json)

    return infoList


def extractBellMTSFile(
    filepath: str, baseDestDirectory: str
) -> Tuple[List[str], List[FileMapping]]:
    return processFile(filepath, "bellMTS", baseDestDirectory, extractBellMTSInfo)


def extractBellMTSDirectory(
    directoryPath: str,
    pageInfo: List[str],
    fileInfo: List[FileMapping],
    baseDestDirectory: str,
):
    iterateDirectory(
        directoryPath, pageInfo, fileInfo, extractBellMTSFile, baseDestDirectory
    )
