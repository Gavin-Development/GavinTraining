import os
import unittest
import json
import shutil

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from GavinCore.models import PreformerIntegration, tfds, tf
from GavinCore.datasets import create_data_objects
from DataParsers.load_data import load_tokenized_data
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


class TestPreformer(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer_path = os.path.join(BASE_DIR, os.path.join('tests/test_files', 'Tokenizer-3'))
        self.tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(self.tokenizer_path)
        self.hparams = {
            'NUM_LAYERS': 4,
            'UNITS': 2048,
            'D_MODEL': 256,
            'NUM_HEADS': 8,
            'DROPOUT': 0.01,
            'MAX_LENGTH': 80,
            'NUM_FEATURES': 128,
            'TOKENIZER': self.tokenizer,
            'MODEL_NAME': "TestPreformer",
            'FLOAT16': False,
            'EPOCHS': 0
        }
        self.config_for_models = self.hparams.copy()
        self.config_for_models = {k.lower(): v for k, v in self.config_for_models.items()}
        self.config_for_models['max_len'] = self.config_for_models['max_length']
        self.config_for_models['name'] = self.config_for_models['model_name']
        self.config_for_models['mixed'] = self.config_for_models['float16']
        self.config_for_models['base_log_dir'] = '../models/'
        del self.config_for_models['max_length'], self.config_for_models['model_name'], self.config_for_models['float16']

    def test_001_model_create(self):
        """Make sure the PreformerIntegration can create a tf.models.Model instance."""
        try:
            base = PreformerIntegration(**self.config_for_models)
            self.assertTrue(hasattr(base, "model"), "Model not created.")
            shutil.rmtree(os.path.join(BASE_DIR, os.path.join('models/', 'TestPreformer')))
        except Exception as e:
            self.fail(f"Model creation failed: {e}")

    def test_002_hparams_return(self):
        """Ensure that hyper-parameters built inside the model, match the users choice."""
        base = PreformerIntegration(**self.config_for_models)
        model_returned_hparams = base.get_hparams()
        shutil.rmtree(os.path.join(BASE_DIR, os.path.join('models/', 'TestPreformer')))
        self.assertDictEqual(model_returned_hparams, self.hparams, f"Model Parameter mismatch.\n"
                                                                   f"Self: {self.hparams}\n"
                                                                   f"Model: {model_returned_hparams}")

    def test_003_model_fit_save(self):
        """Ensure the model trains for at least 1 epoch without an exception."""
        base = PreformerIntegration(**self.config_for_models)
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
                     epochs=1, callbacks=base.get_default_callbacks()[:-1])
        except Exception as e:
            self.fail(f"Model fit failed: {e}")
        base.save_hparams()
        self.assertTrue(os.path.exists('../models/TestPreformer/config/config.json'))
        self.assertTrue(os.path.exists('../models/TestPreformer/tokenizer/TestPreformer_tokenizer.subwords'))
        hparams = self.hparams
        hparams['TOKENIZER'] = os.path.join('../models/TestPreformer',
                                            os.path.join('tokenizer', 'TestPreformer' + '_tokenizer'))
        hparams['EPOCHS'] = hparams['EPOCHS'] + 1
        self.assertEqual(json.load(open('../models/TestPreformer/config/config.json')), hparams)

    def test_004_model_load_fit(self):
        base = PreformerIntegration.load_model('../models/', 'TestPreformer')

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
                     epochs=1, callbacks=base.get_default_callbacks()[:-1])
        except Exception as e:
            self.fail(f"Model fit failed: {e}")

    def test_005_model_callbacks(self):
        base = PreformerIntegration.load_model('../models/', 'TestPreformer')

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
                     epochs=2, callbacks=base.get_default_callbacks())
        except Exception as e:
            self.fail(f"Model fit failed: {e}")

    def test_006_model_predicting(self):
        base = PreformerIntegration.load_model('../models/', 'TestPreformer')

        try:
            reply = base.predict("This is a test.")
            print(f"""\
Prompt: This is a test.
Reply: {reply}""")
        except Exception as e:
            self.fail(f"Model predict failed: {e}")

    def test_007_model_projector_metadata(self):
        try:
            base = PreformerIntegration(**self.config_for_models)
            self.assertTrue(os.path.exists('../models/TestPreformer/metadata.tsv'))
        except Exception as e:
            self.fail(f"Model creation failed: {e}")
