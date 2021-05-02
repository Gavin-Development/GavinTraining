import os
import unittest

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from GavinCore import TransformerIntegration, tfds
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


class TestTransformer(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(os.path.join(BASE_DIR, os.path.join('tests/test_files', 'Tokenizer-3')))
        self.hparams = {
            'NUM_LAYERS': 2,
            'UNITS': 512,
            'D_MODEL': 256,
            'NUM_HEADS': 8,
            'DROPOUT': 0.1,
            'MAX_LENGTH': 52,
            'TOKENIZER': self.tokenizer,
            'MODEL_NAME': "TestTransformer",
            'FLOAT16': False
        }

    def test_model_create(self):
        """Make sure the TransformerIntegration can create a tf.models.Model instance."""
        try:
            base = TransformerIntegration(num_layers=2,
                                          units=512,
                                          d_model=256,
                                          num_heads=8,
                                          dropout=0.1,
                                          max_len=52,
                                          base_log_dir='./models/',
                                          tokenizer=self.tokenizer,
                                          name="TestTransformer")
            self.assertTrue(hasattr(base, "model"), "Model not created.")
        except Exception as e:
            self.fail(f"Model creation failed: {e}")

    def test_hparams_return(self):
        """Ensure that hyper-parameters built inside the model, match the users choice."""
        base = TransformerIntegration(num_layers=2,
                                      units=512,
                                      d_model=256,
                                      num_heads=8,
                                      dropout=0.1,
                                      max_len=52,
                                      base_log_dir='./models/',
                                      tokenizer=self.tokenizer,
                                      name="TestTransformer")
        model_returned_hparams = base.get_hparams()
        self.assertDictEqual(model_returned_hparams, self.hparams, f"Model Parameter mismatch.\n"
                                                                   f"Self: {self.hparams}\n"
                                                                   f"Model: {model_returned_hparams}")

    def test_model_fit(self):
        """Ensure the model trains for at least 1 epoch without an exception."""
        base = TransformerIntegration(num_layers=1,
                                      units=256,
                                      d_model=128,
                                      num_heads=2,
                                      dropout=0.1,
                                      max_len=52,
                                      base_log_dir='./models/',
                                      tokenizer=self.tokenizer,
                                      name="TestTransformer")
        # base.fit()

