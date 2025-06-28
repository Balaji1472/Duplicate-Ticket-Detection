import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import preprocess_text

class TestPreprocessing(unittest.TestCase):
    def test_basic_sentence(self):
        input_text = "I forgot my account password and need help resetting it."
        expected_output = "forget account password need help reset"
        self.assertEqual(preprocess_text(input_text), expected_output)

    def test_stopwords_removal(self):
        input_text = "The internet is not working at all!"
        output = preprocess_text(input_text)
        self.assertNotIn("the", output)
        self.assertNotIn("is", output)

    def test_only_alphabetic(self):
        input_text = "I paid $100 for 2 months!"
        output = preprocess_text(input_text)
        for word in output.split():
            self.assertTrue(word.isalpha())

if __name__ == "__main__":
    unittest.main()
