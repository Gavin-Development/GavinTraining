import os
import unittest
import json

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from GavinCore.models import TransformerIntegration, tfds
from GavinCore.utils import tf
from GavinCore.datasets import DatasetAPICreator
from DataParsers.load_data import load_tokenized_data
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

physical_devices = tf.config.list_physical_devices('GPU')
try:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
except Exception as e:
    print(f"Error on Memory Growth Setting. {e}")
else:
    print("Memory Growth Set to True.")


class TestTransformer(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer_path = os.path.join(BASE_DIR, os.path.join('tests/test_files', 'Tokenizer-3'))
        self.tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(self.tokenizer_path)
        self.max_samples = 10_000
        self.buffer_size = 20_000
        self.batch_size = 32
        self.hparams = {
            'NUM_LAYERS': 1,
            'UNITS': 256,
            'D_MODEL': 128,
            'NUM_HEADS': 2,
            'DROPOUT': 0.1,
            'MAX_LENGTH': 52,
            'TOKENIZER': self.tokenizer,
            'MODEL_NAME': "TestTransformer",
            'FLOAT16': False,
            'EPOCHS': 0,
            'SAVE_FREQ': 'epoch',
            'BATCH_SIZE': self.batch_size
        }
        self.save_freq = 100
        self.config_for_models = self.hparams.copy()
        self.config_for_models = {k.lower(): v for k, v in self.config_for_models.items()}
        self.config_for_models['max_len'] = self.config_for_models['max_length']
        self.config_for_models['name'] = self.config_for_models['model_name']
        self.config_for_models['mixed'] = self.config_for_models['float16']
        self.config_for_models['base_log_dir'] = '../models/'
        del self.config_for_models['max_length'], self.config_for_models['model_name'], self.config_for_models[
            'float16']
        tf.keras.backend.clear_session()  # Reduces the amount of memory this will use.

    def test_001_model_create(self):
        """Make sure the TransformerIntegration can create a tf.models.Model instance."""
        try:
            base = TransformerIntegration(**self.config_for_models)
            self.assertTrue(hasattr(base, "model"), "Model not created.")
        except Exception as e:
            self.fail(f"Model creation failed: {e}")

    def test_002_hparams_return(self):
        """Ensure that hyper-parameters built inside the model, match the users choice."""
        base = TransformerIntegration(**self.config_for_models)
        model_returned_hparams = base.get_hparams()
        self.assertDictEqual(model_returned_hparams, self.hparams, f"Model Parameter mismatch.\n"
                                                                   f"Self: {self.hparams}\n"
                                                                   f"Model: {model_returned_hparams}")

    def test_003_model_fit_save(self):
        """Ensure the model trains for at least 1 epoch without an exception."""
        base = TransformerIntegration(**self.config_for_models)
        questions, answers = load_tokenized_data(max_samples=self.max_samples,
                                                 data_path="D:\\Datasets\\reddit_data\\files\\",
                                                 tokenizer_name="Tokenizer-3",
                                                 s_token=base.start_token,
                                                 e_token=base.end_token, max_len=base.max_len)

        dataset_train, dataset_val = DatasetAPICreator.create_data_objects(questions, answers, buffer_size=self.buffer_size,
                                                                           batch_size=self.batch_size, vocab_size=base.vocab_size)

        try:
            base.fit(training_dataset=dataset_train, validation_dataset=dataset_val,
                     epochs=1, callbacks=base.get_default_callbacks()[:-1])
        except Exception as e:
            self.fail(f"Model fit failed: {e}")
        base.save_hparams()
        self.assertTrue(os.path.exists('../models/TestTransformer/config/config.json'))
        self.assertTrue(os.path.exists('../models/TestTransformer/tokenizer/TestTransformer_tokenizer.subwords'))
        hparams = self.hparams
        hparams['TOKENIZER'] = os.path.join('../models/TestTransformer',
                                            os.path.join('tokenizer', 'TestTransformer' + '_tokenizer'))
        hparams['EPOCHS'] = hparams['EPOCHS'] + 1
        self.assertEqual(json.load(open('../models/TestTransformer/config/config.json')), hparams)

    def test_004_model_load_fit(self):
        base = TransformerIntegration.load_model('../models/', 'TestTransformer')

        questions, answers = load_tokenized_data(max_samples=self.max_samples,
                                                 data_path="D:\\Datasets\\reddit_data\\files\\",
                                                 tokenizer_name="Tokenizer-3",
                                                 s_token=base.start_token,
                                                 e_token=base.end_token, max_len=base.max_len)

        dataset_train, dataset_val = DatasetAPICreator.create_data_objects(questions, answers, buffer_size=self.buffer_size,
                                                                           batch_size=self.batch_size, vocab_size=base.vocab_size)

        try:
            base.fit(training_dataset=dataset_train, validation_dataset=dataset_val,
                     epochs=1)
            base.model.summary()
        except Exception as e:
            self.fail(f"Model fit failed: {e}")

    def test_005_model_projector_metadata(self):
        try:
            base = TransformerIntegration(**self.config_for_models)
            self.assertTrue(os.path.exists('../models/TestTransformer/metadata.tsv'))
        except Exception as e:
            self.fail(f"Model creation failed: {e}")

    def test_006_model_predicting(self):
        base = TransformerIntegration.load_model('../models/', 'TestTransformer')

        try:
            reply = base.predict("This is a test.")
            print(f"""\
        Prompt: This is a test.
        Reply: {reply}""")
        except Exception as e:
            self.fail(f"Model predict failed: {e}")

    def test_007_model_save_freq(self):
        base = TransformerIntegration(**self.config_for_models)
        questions, answers = load_tokenized_data(max_samples=self.max_samples,
                                                 data_path="D:\\Datasets\\reddit_data\\files\\",
                                                 tokenizer_name="Tokenizer-3",
                                                 s_token=base.start_token,
                                                 e_token=base.end_token, max_len=base.max_len)

        dataset_train, dataset_val = DatasetAPICreator.create_data_objects(questions, answers, buffer_size=self.buffer_size,
                                                                           batch_size=self.batch_size, vocab_size=base.vocab_size)
        try:
            base.fit(training_dataset=dataset_train, validation_dataset=dataset_val,
                     epochs=1)
        except Exception as err:
            self.fail(f"Save frequency parameter failed. {err}")
