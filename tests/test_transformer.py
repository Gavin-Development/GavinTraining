import os
import unittest
import json
import shutil

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from GavinCore.models import TransformerIntegration, tfds, tf
from GavinCore.datasets import create_data_objects
from DataParsers.load_data import load_tokenized_data
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


class TestTransformer(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer_path = os.path.join(BASE_DIR, os.path.join('tests/test_files', 'Tokenizer-3'))
        self.tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(self.tokenizer_path)
        self.max_samples = 10_000
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
            'EPOCHS': 0
        }
        self.config_for_models = self.hparams.copy()
        self.config_for_models = {k.lower(): v for k, v in self.config_for_models.items()}
        self.config_for_models['max_len'] = self.config_for_models['max_length']
        self.config_for_models['name'] = self.config_for_models['model_name']
        self.config_for_models['mixed'] = self.config_for_models['float16']
        self.config_for_models['base_log_dir'] = '../models/'
        del self.config_for_models['max_length'], self.config_for_models['model_name'], self.config_for_models[
            'float16']

    def test_001_model_create(self):
        """Make sure the TransformerIntegration can create a tf.models.Model instance."""
        try:
            base = TransformerIntegration(**self.config_for_models)
            self.assertTrue(hasattr(base, "model"), "Model not created.")
            shutil.rmtree(os.path.join(BASE_DIR, os.path.join('models/', 'TestTransformer')))
        except Exception as e:
            self.fail(f"Model creation failed: {e}")

    def test_002_hparams_return(self):
        """Ensure that hyper-parameters built inside the model, match the users choice."""
        base = TransformerIntegration(**self.config_for_models)
        model_returned_hparams = base.get_hparams()
        shutil.rmtree(os.path.join(BASE_DIR, os.path.join('models/', 'TestTransformer')))
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
        hparams['EPOCHS'] = hparams['EPOCHS'] + 1
        self.assertEqual(json.load(open('../models/TestTransformer/config/config.json')), hparams)

    def test_004_model_load_fit(self):
        base = TransformerIntegration.load_model('../models/', 'TestTransformer')

        questions, answers = load_tokenized_data(max_samples=self.max_samples,
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

    def test_005_model_projector_metadata(self):
        try:
            base = TransformerIntegration(**self.config_for_models)
            self.assertTrue(os.path.exists('../models/TestTransformer/metadata.tsv'))
        except Exception as e:
            self.fail(f"Model creation failed: {e}")
