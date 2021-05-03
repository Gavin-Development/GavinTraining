import os
from GavinBackend.GavinCore import tf, tfds
from GavinBackend.GavinCore.models import TransformerIntegration
from GavinBackend.GavinCore.datasets import create_data_objects
from GavinBackend.DataParsers.load_data import load_tokenized_data

LOG_DIR = './bunchOfLogs'
MODEL_NAME = input("Please enter Model_Name: ")
MAX_SAMPLES = int(input("MAX_SAMPLES: "))
BUFFER_SIZE = 20_000
if os.path.exists(os.path.join(LOG_DIR, MODEL_NAME)):
    model = TransformerIntegration.load_model(LOG_DIR, MODEL_NAME)
else:

    BATCH_SIZE = int(input("BATCH_SIZE: "))
    MAX_LENGTH = int(input("MAX_LENGTH: "))
    NUM_LAYERS = int(input("Please enter the number of NUM_LAYERS: "))
    D_MODEL = int(input("Please enter the d_model: "))
    NUM_HEADS = int(input("Please enter the NUM_HEADS: "))
    UNITS = int(input("Please enter the number of units: "))
    DROPOUT = float(input("Please enter the DROPOUT: "))
    EPOCHS = int(input("Please enter the number of epochs: "))
    TOKENIZER_PATH = input("Please enter the Tokenizer Path: ")
    tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(TOKENIZER_PATH)
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
