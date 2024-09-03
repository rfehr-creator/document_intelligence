from typing import List, Tuple
from classes.FileMapping import FileMapping
from classes.JsonKey import Key

from buildDataset.general import (
    extractAccountNumber,
    extractMoney,
    extractPageNumber,
    extractStartEndDate,
    iterateDirectory,
    processFile,
)
from classes.JsonTokenConstants import TokenConstants


def extractPrevBalance(text: str, pattern: str) -> float:
    pass


def extractRBCBankInfo(text: List[str]) -> List[str]:
    rbcBankInfo = []
    for index, page in enumerate(text):
        pageNumber = extractPageNumber(page, r"(\d+)(\s+of\s+)(\d+)")
        startDate, endDate = extractStartEndDate(
            page,
            r"From\s+(\w+\s+\d+,\s+\d+)\s+to\s+(\w+\s+\d+,\s+\d+)",
            r"(\w+)\s+(\d+),\s+(\d+)",
        )
        accountNumber = extractAccountNumber(
            page, r"Your\s+account\s+number:\s+(\d{5}-\d{7})"
        )

        json = {}

        print("-----------------------------")
        print(pageNumber)
        print(startDate)
        print(endDate)
        if pageNumber and startDate and endDate:
            if accountNumber:
                json[Key.accountNumber] = accountNumber
                print(accountNumber)

            json[Key.endDate] = endDate
            json[Key.pageNumber] = pageNumber
            json[Key.startDate] = startDate
            json[Key.type] = TokenConstants.BANK_STATEMENT
            json[Key.vendor] = TokenConstants.RBC

            if index == 0:
                openingBalance = extractMoney(
                    page,
                    r"Your\s+opening\s+balance\D+\d{1,2}\D+\d{4}\D+([\d,]+\.\d{2})",
                )

                closingBalance = extractMoney(
                    page,
                    r"Your\s+closing\s+balance\D+\d{1,2}\D+\d{4}\D+([\d,]+\.\d{2})",
                )
                totalDeposits = extractMoney(
                    page, r"Total\s+deposits\D+([\d,]+\.\d{2})"
                )

                totalWithdrawals = extractMoney(
                    page, r"Total\s+withdrawals\D+([\d,]+\.\d{2})"
                )

                json[Key.openingBalance] = str(openingBalance)

                json[Key.closingBalance] = str(closingBalance)

                json[Key.totalDeposits] = str(totalDeposits)

                json[Key.totalWithdrawals] = str(totalWithdrawals)

                calTotal = openingBalance + totalDeposits - totalWithdrawals

                print("calculated total:", calTotal)
                assert (
                    calTotal == closingBalance
                ), "Closing balance does not match calculated total"

                print("open:", json[Key.openingBalance])
                print("deposits:", json[Key.totalDeposits])
                print("withdrawals:", json[Key.totalWithdrawals])
                print("close:", json[Key.closingBalance])

            # only append if page number, start date, and end date are found
            rbcBankInfo.append(json)

        print("-----------------------------")

    return rbcBankInfo


def extractRBCBankFile(
    filepath: str, baseDestDirectory: str
) -> Tuple[List[str], List[FileMapping]]:
    r, q = processFile(filepath, "rbc_bank", baseDestDirectory, extractRBCBankInfo)
    assert len(r), "No RBC Bank Info found"
    assert len(q), "No RBC Bank File Mapping found"
    return (r, q)


def extractRBCBankDirectory(
    directoryPath: str,
    pageInfo: List[str],
    fileInfo: List[FileMapping],
    baseDestDirectory: str,
):
    iterateDirectory(
        directoryPath, pageInfo, fileInfo, extractRBCBankFile, baseDestDirectory
    )
