import unittest
from DataParsers.load_data import load_tokenized_data


class DataLoad(unittest.TestCase):
    def setUp(self) -> None:
        self.max_samples = 10_000
        self.start_token = [69908]
        self.end_token = [69909]
        self.max_len = 52

    def test_001_legacy_load(self):
        try:
            questions, answers = load_tokenized_data(max_samples=self.max_samples,
                                                     data_path="D:\\Datasets\\reddit_data\\files\\",
                                                     tokenizer_name="Tokenizer-3",
                                                     s_token=self.start_token,
                                                     e_token=self.end_token, legacy=True)
            self.assertEqual(len(questions), self.max_samples//2)
            self.assertEqual(len(answers), self.max_samples//2)
        except Exception as e:
            self.fail(f"Legacy failed: {e}")

    def test_002_CustomPackage_load(self):
        try:
            questions, answers = load_tokenized_data(max_samples=self.max_samples,
                                                     data_path="D:\\Datasets\\reddit_data\\files\\",
                                                     tokenizer_name="Tokenizer-3",
                                                     s_token=self.start_token,
                                                     e_token=self.end_token, max_len=self.max_len)
        except Exception as e:
            self.fail(f"Legacy failed: {e}")

