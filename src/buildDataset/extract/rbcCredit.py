from typing import List, Tuple
import re
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
from classes.JsonTokenConstants import TokenConstants


# RBC Credit Card Statements only have 1 year for the date except if the statement is from December to January
def extractStartEndDate(text: str) -> Tuple[str, str] | Tuple[None, None]:
    dates = re.search(r"FROM\s+(\w+\s+\d+,\s+\d+)\s+TO\s+(\w+\s+\d+,\s+\d+)", text)

    if dates and len(dates.groups()) == 2:
        startDate = extractDate(dates.group(1), r"(\w+)\s+(\d+),\s+(\d+)")
        endDate = extractDate(dates.group(2), r"(\w+)\s+(\d+),\s+(\d+)")
        if startDate and endDate:
            return (startDate, endDate)
    else:
        dates = re.search(r"FROM\s+(\w+\s+\d+)\s+TO\s+(\w+\s+\d+,\s+\d+)", text)
        assert (
            dates and len(dates.groups()) == 2
        ), "Start and end date not found for RBC Credt Card Statement"

        endDate = extractDate(dates.group(2), r"(\w+)\s+(\d+),\s+(\d+)")
        assert endDate, "End date not found for RBC Credit Card Statement"

        startDate = extractDate(dates.group(1), r"(\w+)\s+(\d+)", endDate.get(Key.year))
        assert startDate, "Start date not found for RBC Credit Card Statement"

        return (startDate, endDate)

    return (None, None)


def extractRBCCreditInfo(text: List[str]) -> List[str]:
    rbcBankInfo = []
    for pageText in text:
        json = {}

        pageNumber = extractPageNumber(pageText, r"(\d+)(\s+OF\s+)(\d+)")
        if pageNumber is None:
            # info page for RBC credit card
            json[Key.pageNumber] = "info"
            json[Key.type] = TokenConstants.CREDIT_CARD_STATEMENT
            json[Key.vendor] = TokenConstants.RBC

        else:
            startDate, endDate = extractStartEndDate(pageText)
            assert (
                startDate and endDate
            ), "Start and end date not found for RBC Credit Card"
            accountNumber = extractAccountNumber(
                pageText, r"(\d{4}\D{1}\d{2}\D{8}\d{4})"
            )

            if accountNumber:
                json[Key.accountNumber] = accountNumber.replace("*", "")

            json[Key.endDate] = endDate
            json[Key.pageNumber] = pageNumber
            json[Key.startDate] = startDate
            json[Key.type] = "Credit Card Statement"
            json[Key.vendor] = "RBC Royal Bank"

            if pageNumber == 1:
                prevBalance = extractMinusMoney(
                    pageText,
                    r"Previous\s+(?:Account|Statement)\s+Balance\s+(-?)\$([\d,]+\.\d{2})",
                    "-",
                )
                totalPayments = extractMinusMoney(
                    pageText, r"Payments\s+&\s+credits\s+(\-?)\$([\d,]+\.\d{2})", "-"
                )
                totalPurchases = extractMinusMoney(
                    pageText, r"Purchases\s+&\s+debits\s+(\-?)\$([\d,]+\.\d{2})", "-"
                )
                newBalance = extractMinusMoney(
                    pageText,
                    r"(?:Total\s+Account\s+Balance|NEW\s+BALANCE|CREDIT\s+BALANCE)\s+(\-?)\$([\d,]+\.\d{2})",
                    "-",
                )

                cashAdvances = extractMoney(
                    pageText, r"Cash\s+advances\s+\$([\d,]+\.\d{2})"
                )
                interestCharges = extractMoney(
                    pageText, r"Interest\s+\$([\d,]+\.\d{2})"
                )
                fees = extractMoney(pageText, r"Fees\s+\$([\d,]+\.\d{2})")

                json[Key.cashAdvances] = str(cashAdvances)
                json[Key.fees] = str(fees)
                json[Key.interestCharges] = str(interestCharges)

                json[Key.newBalance] = str(newBalance)
                json[Key.prevBalance] = str(prevBalance)

                json[Key.totalPayments] = str(totalPayments)
                json[Key.totalPurchases] = str(totalPurchases)

                calBalance = (
                    prevBalance
                    + totalPurchases
                    + totalPayments
                    + cashAdvances
                    + interestCharges
                    + fees
                )
                print(prevBalance, totalPurchases, totalPayments, newBalance)
                assert prevBalance is not None, "Previous balance not found"
                assert totalPurchases is not None, "Total purchases not found"
                assert totalPayments is not None, "Total payments not found"
                assert newBalance is not None, "New balance not found"

                print()
                print("Calculated Balance: ", calBalance)
                print("New Balance: ", newBalance)
                assert (
                    calBalance == newBalance
                ), "Calculated balance does not match new balance"

        print("-----------------------------")
        print(pageNumber)
        print(startDate)
        print(endDate)
        print(accountNumber)
        print("-----------------------------")

        rbcBankInfo.append(json)
    return rbcBankInfo


def extractRBCCreditFile(
    filepath: str, baseDestDirectory: str
) -> Tuple[List[str], List[FileMapping]]:
    return processFile(filepath, "rbc_credit", baseDestDirectory, extractRBCCreditInfo)


def extractRBCCreditDirectory(
    directoryPath: str,
    pageInfo: List[str],
    fileInfo: List[FileMapping],
    baseDestDirectory: str,
):
    iterateDirectory(
        directoryPath, pageInfo, fileInfo, extractRBCCreditFile, baseDestDirectory
    )
