import os
import unittest
import json

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from GavinCore.models import TransformerIntegration, tfds, PerformerIntegration
from GavinCore.utils import tf
from GavinCore.datasets import DatasetAPICreator
from GavinCore.metrics import Perplexity, Precision
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


class Metrics(unittest.TestCase):
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
            'BATCH_SIZE': self.batch_size
        }
        self.config_for_models = self.hparams.copy()
        self.config_for_models = {k.lower(): v for k, v in self.config_for_models.items()}
        self.config_for_models['max_len'] = self.config_for_models['max_length']
        self.config_for_models['name'] = self.config_for_models['model_name']
        self.config_for_models['mixed'] = self.config_for_models['float16']
        self.config_for_models['base_log_dir'] = '../models/'
        del self.config_for_models['max_length'], self.config_for_models['model_name'], self.config_for_models[
            'float16']
        tf.keras.backend.clear_session()  # Reduces the amount of memory this will use.

    def test_001_accuracy_metric_transformer(self):
        try:
            base = TransformerIntegration(**self.config_for_models, metrics=['accuracy'])
        except Exception as err:
            self.fail(f"Model creation failed: {err}")
        self.assertTrue(hasattr(base, "model"), "Model not created.")
        questions, answers = load_tokenized_data(max_samples=self.max_samples,
                                                 data_path="D:\\Datasets\\reddit_data\\files\\",
                                                 tokenizer_name="Tokenizer-3",
                                                 s_token=base.start_token,
                                                 e_token=base.end_token, max_len=base.max_len)

        dataset_train, dataset_val = DatasetAPICreator.create_data_objects(questions, answers,
                                                                           buffer_size=self.buffer_size,
                                                                           batch_size=self.batch_size,
                                                                           vocab_size=base.vocab_size)
        try:
            base.fit(training_dataset=dataset_train, validation_dataset=dataset_val,
                     epochs=1)
        except Exception as err:
            self.fail(f"Model Fit failed: {err}")

    def test_002_precision_metric_transformer(self):
        try:
            base = TransformerIntegration(**self.config_for_models, metrics=[Precision(max_len=self.hparams['MAX_LENGTH'], from_logits=True)])
        except Exception as err:
            self.fail(f"Model creation failed: {err}")
        self.assertTrue(hasattr(base, "model"), "Model not created.")
        questions, answers = load_tokenized_data(max_samples=self.max_samples,
                                                 data_path="D:\\Datasets\\reddit_data\\files\\",
                                                 tokenizer_name="Tokenizer-3",
                                                 s_token=base.start_token,
                                                 e_token=base.end_token, max_len=base.max_len)

        dataset_train, dataset_val = DatasetAPICreator.create_data_objects(questions, answers,
                                                                           buffer_size=self.buffer_size,
                                                                           batch_size=self.batch_size,
                                                                           vocab_size=base.vocab_size)
        try:
            base.fit(training_dataset=dataset_train, validation_dataset=dataset_val,
                     epochs=1)
        except Exception as err:
            self.fail(f"Model Fit failed: {err}")

    def test_003_perplexity_metric_transformer(self):
        try:
            base = TransformerIntegration(**self.config_for_models, metrics=[Perplexity(max_len=self.hparams['MAX_LENGTH'])])
        except Exception as err:
            self.fail(f"Model creation failed: {err}")
        self.assertTrue(hasattr(base, "model"), "Model not created.")
        questions, answers = load_tokenized_data(max_samples=self.max_samples,
                                                 data_path="D:\\Datasets\\reddit_data\\files\\",
                                                 tokenizer_name="Tokenizer-3",
                                                 s_token=base.start_token,
                                                 e_token=base.end_token, max_len=base.max_len)

        dataset_train, dataset_val = DatasetAPICreator.create_data_objects(questions, answers,
                                                                           buffer_size=self.buffer_size,
                                                                           batch_size=self.batch_size,
                                                                           vocab_size=base.vocab_size)
        try:
            base.fit(training_dataset=dataset_train, validation_dataset=dataset_val,
                     epochs=1)
        except Exception as err:
            self.fail(f"Model Fit failed: {err}")

    def test_004_accuracy_metric_performer(self):
        try:
            base = PerformerIntegration(**self.config_for_models, metrics=['accuracy'], num_features=128)
        except Exception as err:
            self.fail(f"Model creation failed: {err}")
        self.assertTrue(hasattr(base, "model"), "Model not created.")
        questions, answers = load_tokenized_data(max_samples=self.max_samples,
                                                 data_path="D:\\Datasets\\reddit_data\\files\\",
                                                 tokenizer_name="Tokenizer-3",
                                                 s_token=base.start_token,
                                                 e_token=base.end_token, max_len=base.max_len)

        dataset_train, dataset_val = DatasetAPICreator.create_data_objects(questions, answers,
                                                                           buffer_size=self.buffer_size,
                                                                           batch_size=self.batch_size,
                                                                           vocab_size=base.vocab_size)
        try:
            base.fit(training_dataset=dataset_train, validation_dataset=dataset_val,
                     epochs=1)
        except Exception as err:
            self.fail(f"Model Fit failed: {err}")

    def test_005_precision_metric_performer(self):
        try:
            base = PerformerIntegration(**self.config_for_models, metrics=[Precision(max_len=self.hparams['MAX_LENGTH'], from_logits=True)], num_features=128)
        except Exception as err:
            self.fail(f"Model creation failed: {err}")
        self.assertTrue(hasattr(base, "model"), "Model not created.")
        questions, answers = load_tokenized_data(max_samples=self.max_samples,
                                                 data_path="D:\\Datasets\\reddit_data\\files\\",
                                                 tokenizer_name="Tokenizer-3",
                                                 s_token=base.start_token,
                                                 e_token=base.end_token, max_len=base.max_len)

        dataset_train, dataset_val = DatasetAPICreator.create_data_objects(questions, answers,
                                                                           buffer_size=self.buffer_size,
                                                                           batch_size=self.batch_size,
                                                                           vocab_size=base.vocab_size)
        try:
            base.fit(training_dataset=dataset_train, validation_dataset=dataset_val,
                     epochs=1)
        except Exception as err:
            self.fail(f"Model Fit failed: {err}")

    def test_006_perplexity_metric_performer(self):
        try:
            base = PerformerIntegration(**self.config_for_models, metrics=[Perplexity(max_len=self.hparams['MAX_LENGTH'])], num_features=128)
        except Exception as err:
            self.fail(f"Model creation failed: {err}")
        self.assertTrue(hasattr(base, "model"), "Model not created.")
        questions, answers = load_tokenized_data(max_samples=self.max_samples,
                                                 data_path="D:\\Datasets\\reddit_data\\files\\",
                                                 tokenizer_name="Tokenizer-3",
                                                 s_token=base.start_token,
                                                 e_token=base.end_token, max_len=base.max_len)

        dataset_train, dataset_val = DatasetAPICreator.create_data_objects(questions, answers,
                                                                           buffer_size=self.buffer_size,
                                                                           batch_size=self.batch_size,
                                                                           vocab_size=base.vocab_size)
        try:
            base.fit(training_dataset=dataset_train, validation_dataset=dataset_val,
                     epochs=1)
        except Exception as err:
            self.fail(f"Model Fit failed: {err}")
