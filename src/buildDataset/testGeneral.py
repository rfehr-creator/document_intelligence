import unittest
from buildDataset.general import longMonthToNumber, readPdfText


class TestReadPdfText(unittest.TestCase):
    def test_readPdfText(self):
        # Assuming you have a sample PDF file named 'sample.pdf' for testing
        filename = (
            "../data/original/rbcBankStatements/Chequing Statement-3139 2021-01-08.pdf"
        )

        # Call the function
        result = readPdfText(filename)

        # Check if the result matches the expected output
        self.assertTrue(
            result[0].__contains__("From December 8, 2020 to January 8, 2021")
        )


class TestLongMonthToNumber(unittest.TestCase):
    def test_january(self):
        self.assertEqual(longMonthToNumber("January"), 1)

    def test_february(self):
        self.assertEqual(longMonthToNumber("February"), 2)

    def test_march(self):
        self.assertEqual(longMonthToNumber("March"), 3)

    def test_april(self):
        self.assertEqual(longMonthToNumber("April"), 4)

    def test_may(self):
        self.assertEqual(longMonthToNumber("May"), 5)

    def test_june(self):
        self.assertEqual(longMonthToNumber("June"), 6)

    def test_july(self):
        self.assertEqual(longMonthToNumber("July"), 7)

    def test_august(self):
        self.assertEqual(longMonthToNumber("August"), 8)

    def test_september(self):
        self.assertEqual(longMonthToNumber("September"), 9)

    def test_october(self):
        self.assertEqual(longMonthToNumber("October"), 10)

    def test_november(self):
        self.assertEqual(longMonthToNumber("November"), 11)

    def test_december(self):
        self.assertEqual(longMonthToNumber("December"), 12)

    def test_invalid_month(self):
        with self.assertRaises(KeyError):
            longMonthToNumber("InvalidMonth")


if __name__ == "__main__":
    unittest.main()
