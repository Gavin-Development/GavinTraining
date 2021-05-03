import os
import tensorflow as tf
import tensorflow_datasets as tfds
from GavinBackend.GavinCore.models import TransformerIntegration
from GavinBackend.GavinCore.datasets import create_data_objects
from GavinBackend.DataParsers.load_data import load_tokenized_data

LOG_DIR = './bunchOfLogs'
MODEL_NAME = input("Please enter Model_Name: ")
MAX_SAMPLES = int(input("MAX_SAMPLES: "))
BATCH_SIZE = int(input("BATCH_SIZE: "))
BUFFER_SIZE = int(input("BUFFER_SIZE: "))
TOKENIZER_PATH = input("TOKENIZER_PATH: ")
EPOCHS = int(input("EPOCHS: "))
tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(TOKENIZER_PATH)
if os.path.exists(os.path.join(LOG_DIR, MODEL_NAME)):
    model = TransformerIntegration.load_model(LOG_DIR, MODEL_NAME)
    questions, answers = load_tokenized_data(max_samples=MAX_SAMPLES,
                                             data_path="D:\\Datasets\\reddit_data\\files\\",
                                             tokenizer_name=os.path.basename(TOKENIZER_PATH),
                                             s_token=model.start_token,
                                             e_token=model.end_token, )
    questions = tf.keras.preprocessing.sequence.pad_sequences(questions, maxlen=model.max_len, padding='post')
    answers = tf.keras.preprocessing.sequence.pad_sequences(answers, maxlen=model.max_len, padding='post')
    dataset_train, dataset_val = create_data_objects(questions, answers, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE)
    model.fit(dataset_train, validation_dataset=dataset_val, epochs=EPOCHS)
else:
    MAX_LENGTH = int(input("MAX_LENGTH: "))
    NUM_LAYERS = int(input("NUM_LAYERS: "))
    D_MODEL = int(input("D_MODEL: "))
    NUM_HEADS = int(input("NUM_HEADS: "))
    UNITS = int(input("UNITS: "))
    DROPOUT = float(input("DROPOUT: "))
    model = TransformerIntegration(num_layers=NUM_LAYERS, units=UNITS, d_model=D_MODEL,
                                   num_heads=NUM_HEADS, base_log_dir=LOG_DIR, dropout=DROPOUT,
                                   max_len=MAX_LENGTH, tokenizer=TOKENIZER_PATH)
    questions, answers = load_tokenized_data(max_samples=MAX_SAMPLES,
                                             data_path="D:\\Datasets\\reddit_data\\files\\",
                                             tokenizer_name=os.path.basename(TOKENIZER_PATH),
                                             s_token=model.start_token,
                                             e_token=model.end_token, )
    questions = tf.keras.preprocessing.sequence.pad_sequences(questions, maxlen=model.max_len, padding='post')
    answers = tf.keras.preprocessing.sequence.pad_sequences(answers, maxlen=model.max_len, padding='post')
    dataset_train, dataset_val = create_data_objects(questions, answers, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE)
    model.fit(dataset_train, validation_dataset=dataset_val, epochs=EPOCHS)
