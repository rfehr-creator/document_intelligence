import os
import sys

# Add the parent directory of 'classes' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from buildDataset.extract.bellMTS import extractBellMTSFile
from extract.extract import Extract
from buildDataset.general import readPdfText

rbcStatement = "/home/roland/ml_playground-1/pytorch/document_intelligence/data/original/rbcBankStatements/Chequing Statement-3139 2021-01-08.pdf"
# info = extractRBCBankFile(rbcStatement)
# rbc = "../data/original/rbcBankStatements"

# cc = "../data/original/rbcCreditCardStatements/Visa Statement-1214 2023-06-21.pdf"
cc = "../data/original/rbcCreditCardStatements/Visa Statement-1214 2021-01-21.pdf"
hydro = "/home/roland/ml_playground-1/pytorch/document_intelligence/data/original/hydro/2022-08.pdf"
hydro = "/home/roland/ml_playground-1/pytorch/document_intelligence/data/original/hydro/2024-06.pdf"
saving = "/home/roland/ml_playground-1/pytorch/document_intelligence/data/original/rbcSavings/Savings Statement-4050 2018-11-08.pdf"
bellMTS = "/home/roland/ml_playground-1/pytorch/document_intelligence/data/original/bellMTS/2022_08_10.pdf"
bellMTS = "/home/roland/ml_playground-1/pytorch/document_intelligence/data/original/bellMTS/2023_05_10.pdf"
# text = readPdfText(cc)
# print(text)

# info = extractRBCCreditFile(cc)
# for i in info:
#     print(i)

if len(sys.argv) == 2:
    if sys.argv[1] == "mem":
        Extract().run(memorize=True)
    else:
        Extract().run()
else:
    text = readPdfText(bellMTS)

    for t in text:
        print(t)
        print("------------------------- end page -------------------------\n")
    
    extractBellMTSFile(bellMTS, "hello")
    

