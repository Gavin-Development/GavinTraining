import os
from GavinBackend.GavinCore import tf, tfds
from GavinBackend.GavinCore.models import TransformerIntegration
from GavinBackend.GavinCore.datasets import create_data_objects
from GavinBackend.DataParsers.load_data import load_tokenized_data

LOG_DIR = './bunchOfLogs'
MODEL_NAME = input("Please enter Model_Name: ")
if os.path.exists(os.path.join(LOG_DIR, MODEL_NAME)):
    model = TransformerIntegration.load_model(LOG_DIR, MODEL_NAME)
else:
    MAX_SAMPLES = int(input("MAX_SAMPLES: "))
    BATCH_SIZE = int(input("BATCH_SIZE(32): "))
    BUFFER_SIZE = 20_000
    MAX_LENGTH = 40 + 2
    NUM_LAYERS = int(input("Please enter the number of NUM_LAYERS(4): "))
    D_MODEL = int(input("Please enter the d_model(256): "))
    NUM_HEADS = int(input("Please enter the NUM_HEADS(8): "))
    UNITS = int(input("Please enter the number of units(512): "))
    DROPOUT = float(input("Please enter the DROPOUT(0.175): "))
    EPOCHS = int(input("Please enter the number of epochs(15): "))
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
