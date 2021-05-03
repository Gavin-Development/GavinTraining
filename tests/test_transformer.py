import os
import unittest
import json

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from GavinCore import TransformerIntegration, tfds, tf
from GavinCore.datasets import create_data_objects
from DataParsers.load_data import load_tokenized_data
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


class TestTransformer(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer_path = os.path.join(BASE_DIR, os.path.join('tests/test_files', 'Tokenizer-3'))
        self.tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(self.tokenizer_path)
        self.hparams = {
            'NUM_LAYERS': 1,
            'UNITS': 256,
            'D_MODEL': 128,
            'NUM_HEADS': 2,
            'DROPOUT': 0.1,
            'MAX_LENGTH': 52,
            'TOKENIZER': self.tokenizer,
            'MODEL_NAME': "TestTransformer",
            'FLOAT16': False
        }

    def test_model_create(self):
        """Make sure the TransformerIntegration can create a tf.models.Model instance."""
        try:
            base = TransformerIntegration(num_layers=1,
                                          units=256,
                                          d_model=128,
                                          num_heads=2,
                                          dropout=0.1,
                                          max_len=52,
                                          base_log_dir='../models/',
                                          tokenizer=self.tokenizer,
                                          name="TestTransformer")
            self.assertTrue(hasattr(base, "model"), "Model not created.")
        except Exception as e:
            self.fail(f"Model creation failed: {e}")

    def test_hparams_return(self):
        """Ensure that hyper-parameters built inside the model, match the users choice."""
        base = TransformerIntegration(num_layers=1,
                                      units=256,
                                      d_model=128,
                                      num_heads=2,
                                      dropout=0.1,
                                      max_len=52,
                                      base_log_dir='../models/',
                                      tokenizer=self.tokenizer,
                                      name="TestTransformer")
        model_returned_hparams = base.get_hparams()
        self.assertDictEqual(model_returned_hparams, self.hparams, f"Model Parameter mismatch.\n"
                                                                   f"Self: {self.hparams}\n"
                                                                   f"Model: {model_returned_hparams}")

    def test_model_fit_and_save(self):
        """Ensure the model trains for at least 1 epoch without an exception."""
        base = TransformerIntegration(num_layers=1,
                                      units=256,
                                      d_model=128,
                                      num_heads=2,
                                      dropout=0.1,
                                      max_len=52,
                                      base_log_dir='../models/',
                                      tokenizer=self.tokenizer,
                                      name="TestTransformer")
        questions, answers = load_tokenized_data(max_samples=10_000,
                                                 data_path="D:\\Datasets\\reddit_data\\files\\",
                                                 tokenizer_name="Tokenizer-3",
                                                 s_token=base.start_token,
                                                 e_token=base.end_token, )
        questions = tf.keras.preprocessing.sequence.pad_sequences(questions, maxlen=base.max_len, padding='post')
        answers = tf.keras.preprocessing.sequence.pad_sequences(answers, maxlen=base.max_len, padding='post')
        dataset_train, dataset_val = create_data_objects(questions, answers, buffer_size=20_000, batch_size=32)

        try:
            base.fit(training_dataset=dataset_train, validation_dataset=dataset_val,
                     epochs=1)
        except Exception as e:
            self.fail(f"Model fit failed: {e}")
        base.save_hparams()
        self.assertTrue(os.path.exists('../models/TestTransformer/config/config.json'))
        self.assertTrue(os.path.exists('../models/TestTransformer/tokenizer/TestTransformer_tokenizer.subwords'))
        hparams = self.hparams
        hparams['TOKENIZER'] = os.path.join('../models/TestTransformer', os.path.join('tokenizer', 'TestTransformer' + '_tokenizer'))
        self.assertEqual(json.load(open('../models/TestTransformer/config/config.json')), hparams)
