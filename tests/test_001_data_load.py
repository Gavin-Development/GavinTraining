import unittest
import numpy as np
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
        except Exception as e:
            self.fail(f"Legacy failed: {e}")
        self.assertEqual(len(questions), self.max_samples//2)
        self.assertEqual(len(answers), self.max_samples//2)
        self.assertEqual(type(answers), list)
        self.assertEqual(type(questions), list)

    def test_002_CustomPackage_load(self):
        try:
            questions, answers = load_tokenized_data(max_samples=self.max_samples,
                                                     data_path="D:\\Datasets\\reddit_data\\files\\",
                                                     tokenizer_name="Tokenizer-3",
                                                     s_token=self.start_token,
                                                     e_token=self.end_token, max_len=self.max_len)
        except Exception as e:
            self.fail(f"Custom Load failed: {e}")
        self.assertEqual(len(questions), self.max_samples // 2)
        self.assertEqual(len(answers), self.max_samples // 2)
        self.assertEqual(np.ndarray, type(questions),
                         msg=f"type of questions is not of type {np.ndarray} but of type {type(questions)}")
        self.assertEqual(np.ndarray, type(answers),
                         msg=f"type of answers is not of type {np.ndarray} but of type {type(answers)}")
        self.assertEqual((self.max_samples // 2, self.max_len), np.shape(questions),
                         msg=f"Questions is not of size {(self.max_samples // 2, self.max_len)} but of size {np.size(questions)}")
        self.assertEqual((self.max_samples // 2, self.max_len), np.shape(answers),
                         msg=f"Answers is not of size {(self.max_samples // 2, self.max_len)} but of size {np.size(answers)}")
