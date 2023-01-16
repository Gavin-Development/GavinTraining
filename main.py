import os
import json
import shutil
import glob
import numpy as np

os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

from GavinBackend.GavinCore.models import TransformerIntegration, tf, tfds, PerformerIntegration, FNetIntegration, PreTrainedEmbeddingTransformerIntegration, \
    RotaryTransformerIntegration
from GavinBackend.GavinCore.datasets import DatasetAPICreator, DatasetDirectFromFileAPICreator
from GavinBackend.GavinCore.load_data import load_tokenized_data
from GavinBackend.GavinCore.callbacks import PredictCallback


def get_embedding_idx(embedding_path):
    embedding_idx = {}
    with open(embedding_path, 'r', encoding='utf-8') as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embedding_idx[word] = coefs
    return embedding_idx


def get_embedding_matrix(embedding_idx, i_tokenizer: tfds.deprecated.text.SubwordTextEncoder):
    i_dff = int(embedding_idx.get(list(embedding_idx.keys())[0]).shape[0])
    embedding_matrix = np.zeros((len(i_tokenizer.subwords) + 1, i_dff))
    for i, word in enumerate(i_tokenizer.subwords):
        embedding_vector = embedding_idx.get(word)
        if embedding_vector is not None and embedding_vector.shape[0] == i_dff:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix, i_dff


def get_train_data(max_samples, dataset_path, file_name, t_model, buffer_size, batch_size, python_legacy=False, cpp_legacy=False,
                   use_memory_loaders=True):
    if python_legacy or use_memory_loaders:
        questions, answers = load_tokenized_data(max_samples=max_samples,
                                                 data_path=dataset_path,
                                                 filename=file_name,
                                                 s_token=t_model.start_token,
                                                 e_token=t_model.end_token, max_len=t_model.max_len,
                                                 python_legacy=python_legacy,
                                                 cpp_legacy=cpp_legacy)

        questions = tf.keras.preprocessing.sequence.pad_sequences(questions, maxlen=t_model.max_len, padding='post')
        answers = tf.keras.preprocessing.sequence.pad_sequences(answers, maxlen=t_model.max_len, padding='post')
        d_t, d_v = DatasetAPICreator.create_data_objects(questions, answers, buffer_size=buffer_size,
                                                         batch_size=batch_size,
                                                         vocab_size=t_model.vocab_size)
    else:
        path_to = os.path.join(DATASET_PATH, "{}-{}.BIN")
        # noinspection StrFormat
        d_t, d_v = DatasetDirectFromFileAPICreator.create_data_objects(questions_file=path_to.format(file_name, "from"),
                                                                       answers_file=path_to.format(file_name, "to"),
                                                                       buffer_size=buffer_size,
                                                                       batch_size=batch_size,
                                                                       vocab_size=t_model.vocab_size,
                                                                       max_length=t_model.max_len,
                                                                       number_of_samples=max_samples,
                                                                       start_token=t_model.start_token[0],
                                                                       end_token=t_model.end_token[0],
                                                                       padding_value=0)

    return d_t, d_v


physical_devices = tf.config.list_physical_devices('GPU')
try:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
except Exception as e:
    print(f"Error on Memory Growth Setting. {e}")
else:
    print("Memory Growth Set to True.")

if not os.path.exists("bunchOfLogs"):
    os.mkdir("./bunchOfLogs/")

PYTHON_LEGACY = False
CPP_LEGACY = False

DATASET_PATH = input("Please enter dataset path: ")
PYTHON_LEGACY = True if "https" in DATASET_PATH else PYTHON_LEGACY
if not PYTHON_LEGACY:
    print("Files in dataset path:")
    files = glob.glob(os.path.join(DATASET_PATH, "*.BIN"))
    files = list(set([os.path.basename(file).split('.')[0].replace('-from', '').replace('-to', '') for file in files]))  # Quick and dirty
    for i, file in enumerate(files):
        print(f"{i+1}: {file}")
dataset_file_name = input("Please enter dataset file name: ")


MODEL_TYPE = input("Please enter a Model Type [`performer`, `transformer`, `fnet`, `pretrained`, `rotary`]: ")

MIXED = bool(input("Enable mixed precision y/n: ").lower() in ["n", "no"])
if MIXED:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

if MODEL_TYPE.lower() == "performer":
    MODEL_TYPE = PerformerIntegration
elif MODEL_TYPE.lower() == "transformer":
    MODEL_TYPE = TransformerIntegration
elif MODEL_TYPE.lower() == "fnet":
    MODEL_TYPE = FNetIntegration
elif MODEL_TYPE.lower() == "pretrained":
    MODEL_TYPE = PreTrainedEmbeddingTransformerIntegration
elif MODEL_TYPE.lower() == "rotary":
    MODEL_TYPE = RotaryTransformerIntegration
else:
    print("Invalid model type. Quitting")
    quit()
LOG_DIR = './bunchOfLogs'
MODEL_NAME = input("Please enter Model_Name: ")

EMBEDDING_FILE = None
if os.path.exists(os.path.join(LOG_DIR, MODEL_NAME)):
    try:
        metadata = json.load(
            open(os.path.join(LOG_DIR, os.path.join(MODEL_NAME, os.path.join('config/', 'metadata.json')))))
        choice = input(f"Do you want to edit the metadata ({metadata})? y/n: ")
        if choice == "y":
            MAX_SAMPLES = int(input("MAX_SAMPLES: "))
            BATCH_SIZE = int(input("BATCH_SIZE: "))
            BUFFER_SIZE = int(input("BUFFER_SIZE: "))
        else:
            MAX_SAMPLES = metadata['MAX_SAMPLES']
            BATCH_SIZE = metadata['BATCH_SIZE']
            BUFFER_SIZE = metadata['BUFFER_SIZE']
    except FileNotFoundError:
        answer = input("No metadata found. Would you like to delete the model dir? y/n: ")
        if answer.strip() == "y":
            shutil.rmtree(os.path.join(LOG_DIR, MODEL_NAME))
        MAX_SAMPLES = None
        BATCH_SIZE = None
        BUFFER_SIZE = None
        quit()
else:
    MAX_SAMPLES = int(input("MAX_SAMPLES: "))
    BATCH_SIZE = int(input("BATCH_SIZE: "))
    BUFFER_SIZE = int(input("BUFFER_SIZE: "))
    if MODEL_TYPE == PreTrainedEmbeddingTransformerIntegration:
        EMBEDDING_FILE = input("Please enter embedding file path: ")
        if not os.path.exists(EMBEDDING_FILE):
            print("Invalid embedding file path. Quitting")
            quit(-1)

TOKENIZER_PATH = input("TOKENIZER_PATH: ")
EPOCHS = int(input("EPOCHS: "))
tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(TOKENIZER_PATH)
metadata = {'MAX_SAMPLES': MAX_SAMPLES, 'BATCH_SIZE': BATCH_SIZE, 'BUFFER_SIZE': BUFFER_SIZE}

if not os.path.exists(os.path.join(DATASET_PATH, dataset_file_name+"-from.BIN")) or not os.path.exists(os.path.join(DATASET_PATH, dataset_file_name+"-to.BIN")):
    print("Dataset files not found. Quitting")
    print(f"Expected files: {os.path.join(DATASET_PATH, dataset_file_name+'-from.BIN')} and {os.path.join(DATASET_PATH, dataset_file_name+'-to.BIN')}")
    quit(-1)
if os.path.exists(os.path.join(LOG_DIR, MODEL_NAME)):
    model = MODEL_TYPE.load_model(LOG_DIR, MODEL_NAME)
    model.metadata = metadata
else:
    MAX_LENGTH = int(input("MAX_LENGTH: "))
    NUM_LAYERS = int(input("NUM_LAYERS: "))
    D_MODEL = int(input("D_MODEL: "))
    NUM_HEADS = int(input("NUM_HEADS: "))
    UNITS = int(input("UNITS: "))
    DROPOUT = float(input("DROPOUT: "))
    SAVE_FREQ = input("Press Enter to save by epoch, or type a number to save by batch: ")
    if SAVE_FREQ == "\n" or SAVE_FREQ == "":
        SAVE_FREQ = 'epoch'
    else:
        SAVE_FREQ = int(SAVE_FREQ)

    if MODEL_TYPE in [TransformerIntegration, RotaryTransformerIntegration]:
        model = MODEL_TYPE(num_layers=NUM_LAYERS, units=UNITS, d_model=D_MODEL,
                           num_heads=NUM_HEADS, base_log_dir=LOG_DIR, dropout=DROPOUT,
                           max_len=MAX_LENGTH, tokenizer=tokenizer, name=MODEL_NAME,
                           metadata=metadata,
                           save_freq=SAVE_FREQ, batch_size=BATCH_SIZE,
                           mixed=MIXED)
    elif MODEL_TYPE == FNetIntegration:
        model = MODEL_TYPE(num_layers=NUM_LAYERS, units=UNITS, d_model=D_MODEL,
                           num_heads=NUM_HEADS, base_log_dir=LOG_DIR, dropout=DROPOUT,
                           max_len=MAX_LENGTH, tokenizer=tokenizer, name=MODEL_NAME,
                           metadata=metadata,
                           save_freq=SAVE_FREQ, batch_size=BATCH_SIZE,
                           mixed=MIXED)
    elif MODEL_TYPE == PreTrainedEmbeddingTransformerIntegration:
        matrix, dff = get_embedding_matrix(get_embedding_idx(EMBEDDING_FILE), tokenizer)
        print(f"You selected {D_MODEL} however a value of {dff} was used for D_MODEL because the embedding file was {dff} in size.")
        model = MODEL_TYPE(num_layers=NUM_LAYERS, units=UNITS, d_model=dff,
                           num_heads=NUM_HEADS, base_log_dir=LOG_DIR, dropout=DROPOUT,
                           max_len=MAX_LENGTH, tokenizer=tokenizer, name=MODEL_NAME,
                           metadata=metadata,
                           save_freq=SAVE_FREQ, batch_size=BATCH_SIZE, embedding_matrix=matrix,
                           mixed=MIXED)
    elif MODEL_TYPE == PerformerIntegration:
        NUM_FEATURES = int(input("RANDOM_FEATURES: "))
        model = MODEL_TYPE(num_layers=NUM_LAYERS, units=UNITS, d_model=D_MODEL,
                           num_heads=NUM_HEADS, base_log_dir=LOG_DIR, dropout=DROPOUT,
                           max_len=MAX_LENGTH, tokenizer=tokenizer, name=MODEL_NAME,
                           save_freq=SAVE_FREQ, batch_size=BATCH_SIZE,
                           num_features=NUM_FEATURES,
                           mixed=MIXED)
    else:
        model = None
        quit(-1)

memory_load = bool(input("Would you like to stream samples direct from file? y/n: ").lower() in ["n", "no"])

dataset_train, dataset_val = get_train_data(max_samples=MAX_SAMPLES, dataset_path=DATASET_PATH, file_name=dataset_file_name, t_model=model,
                                            buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, python_legacy=PYTHON_LEGACY, cpp_legacy=CPP_LEGACY,
                                            use_memory_loaders=memory_load)

callbacks = model.get_default_callbacks()
callbacks.pop(1)
callbacks.insert(1, tf.keras.callbacks.TensorBoard(log_dir=model.log_dir, update_freq=model.save_freq,
                                                   embeddings_metadata=os.path.join(model.log_dir, "metadata.tsv"),
                                                   profile_batch=(100, 110), embeddings_freq=5))
callbacks.pop(2)
callbacks.append(PredictCallback(tokenizer=tokenizer, start_token=model.start_token, end_token=model.end_token,
                                 max_length=model.max_len, log_dir=model.log_dir, update_freq=model.save_freq,
                                 wrapper_model=model))
callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True))

model.fit(dataset_train, validation_dataset=dataset_val, epochs=EPOCHS, callbacks=callbacks)
model.model.summary()
