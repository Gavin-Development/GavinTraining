import os
import json
import platform
import shutil

os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

from GavinBackend.GavinCore.models import TransformerIntegration, tf, tfds, PerformerIntegration, FNetIntegration
from GavinBackend.GavinCore.datasets import DatasetAPICreator
from GavinBackend.DataParsers.load_data import load_tokenized_data
from GavinBackend.GavinCore.metrics import Perplexity
from GavinBackend.GavinCore.callbacks import PredictCallback

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

PYTHON_LEGACY = False if "windows" in platform.system().lower() else True
CPP_LEGACY = False
DATASET_PATH = input("Please enter dataset path: ")
MODEL_TYPE = input("Please enter a Model Type [`performer`, `transformer`, `fnet`]: ")
if MODEL_TYPE.lower() == "performer":
    MODEL_TYPE = PerformerIntegration
elif MODEL_TYPE.lower() == "transformer":
    MODEL_TYPE = TransformerIntegration
elif MODEL_TYPE.lower() == "fnet":
    MODEL_TYPE = FNetIntegration
else:
    print("Invalid model type. Quitting")
    quit()
LOG_DIR = './bunchOfLogs'
MODEL_NAME = input("Please enter Model_Name: ")
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
        quit()
else:
    MAX_SAMPLES = int(input("MAX_SAMPLES: "))
    BATCH_SIZE = int(input("BATCH_SIZE: "))
    BUFFER_SIZE = int(input("BUFFER_SIZE: "))
TOKENIZER_PATH = input("TOKENIZER_PATH: ")
EPOCHS = int(input("EPOCHS: "))
tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(TOKENIZER_PATH)
metadata = {'MAX_SAMPLES': MAX_SAMPLES, 'BATCH_SIZE': BATCH_SIZE, 'BUFFER_SIZE': BUFFER_SIZE}
dataset_file_name = "Tokenizer-3"
if os.path.exists(os.path.join(LOG_DIR, MODEL_NAME)):
    model = MODEL_TYPE.load_model(LOG_DIR, MODEL_NAME)
    with model.strategy.scope():
        model.metrics = [tf.keras.metrics.SparseCategoricalAccuracy(),
                         Perplexity(max_len=model.max_len, vocab_size=model.vocab_size)]
    model.metadata = metadata
    questions, answers = load_tokenized_data(max_samples=MAX_SAMPLES,
                                             data_path=DATASET_PATH,
                                             filename=dataset_file_name,
                                             s_token=model.start_token,
                                             e_token=model.end_token, max_len=model.max_len,
                                             python_legacy=PYTHON_LEGACY,
                                             cpp_legacy=CPP_LEGACY)
    if PYTHON_LEGACY:
        questions = tf.keras.preprocessing.sequence.pad_sequences(questions, maxlen=model.max_len, padding='post')
        answers = tf.keras.preprocessing.sequence.pad_sequences(answers, maxlen=model.max_len, padding='post')
    dataset_train, dataset_val = DatasetAPICreator.create_data_objects(questions, answers, buffer_size=BUFFER_SIZE,
                                                                       batch_size=BATCH_SIZE,
                                                                       vocab_size=model.vocab_size)
    callbacks = model.get_default_callbacks()
    callbacks.pop(1)
    callbacks.insert(1, tf.keras.callbacks.TensorBoard(log_dir=model.log_dir, update_freq=model.save_freq,
                                                       embeddings_metadata=os.path.join(model.log_dir, "metadata.tsv"),
                                                       profile_batch=(100, 110)))
    callbacks.pop(len(callbacks) - 1)
    callbacks.append(PredictCallback(tokenizer=tokenizer, start_token=model.start_token, end_token=model.end_token,
                                     max_length=model.max_len, log_dir=model.log_dir, update_freq=model.save_freq,
                                     wrapper_model=model))
    model.fit(dataset_train, validation_dataset=dataset_val, epochs=EPOCHS, callbacks=callbacks)
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

    if MODEL_TYPE == TransformerIntegration:
        model = MODEL_TYPE(num_layers=NUM_LAYERS, units=UNITS, d_model=D_MODEL,
                           num_heads=NUM_HEADS, base_log_dir=LOG_DIR, dropout=DROPOUT,
                           max_len=MAX_LENGTH, tokenizer=tokenizer, name=MODEL_NAME,
                           metadata=metadata,
                           save_freq=SAVE_FREQ, batch_size=BATCH_SIZE)
        with model.strategy.scope():
            model.metrics.append(Perplexity(max_len=MAX_LENGTH, vocab_size=tokenizer.vocab_size))
    elif MODEL_TYPE == FNetIntegration:
        model = MODEL_TYPE(num_layers=NUM_LAYERS, units=UNITS, d_model=D_MODEL,
                           num_heads=NUM_HEADS, base_log_dir=LOG_DIR, dropout=DROPOUT,
                           max_len=MAX_LENGTH, tokenizer=tokenizer, name=MODEL_NAME,
                           metadata=metadata,
                           save_freq=SAVE_FREQ, batch_size=BATCH_SIZE)
        with model.strategy.scope():
            model.metrics.append(Perplexity(max_len=MAX_LENGTH, vocab_size=tokenizer.vocab_size))
    else:
        NUM_FEATURES = int(input("RANDOM_FEATURES: "))
        model = MODEL_TYPE(num_layers=NUM_LAYERS, units=UNITS, d_model=D_MODEL,
                           num_heads=NUM_HEADS, base_log_dir=LOG_DIR, dropout=DROPOUT,
                           max_len=MAX_LENGTH, tokenizer=tokenizer, name=MODEL_NAME,
                           save_freq=SAVE_FREQ, batch_size=BATCH_SIZE,
                           num_features=NUM_FEATURES)
        with model.strategy.scope():
            model.metrics.append(Perplexity(max_len=MAX_LENGTH, vocab_size=tokenizer.vocab_size))
    questions, answers = load_tokenized_data(max_samples=MAX_SAMPLES,
                                             data_path=DATASET_PATH,
                                             filename=dataset_file_name,
                                             s_token=model.start_token,
                                             e_token=model.end_token, max_len=MAX_LENGTH,
                                             python_legacy=PYTHON_LEGACY,
                                             cpp_legacy=CPP_LEGACY)
    if PYTHON_LEGACY:
        questions = tf.keras.preprocessing.sequence.pad_sequences(questions, maxlen=model.max_len, padding='post')
        answers = tf.keras.preprocessing.sequence.pad_sequences(answers, maxlen=model.max_len, padding='post')
    dataset_train, dataset_val = DatasetAPICreator.create_data_objects(questions, answers, buffer_size=BUFFER_SIZE,
                                                                       batch_size=BATCH_SIZE,
                                                                       vocab_size=model.vocab_size)

    callbacks = model.get_default_callbacks()
    callbacks.pop(1)
    callbacks.insert(1, tf.keras.callbacks.TensorBoard(log_dir=model.log_dir, update_freq=model.save_freq,
                                                       embeddings_metadata=os.path.join(model.log_dir, "metadata.tsv"),
                                                       profile_batch=(100, 110)))
    callbacks.pop(len(callbacks) - 1)
    callbacks.append(PredictCallback(tokenizer=tokenizer, start_token=model.start_token, end_token=model.end_token,
                                     max_length=model.max_len, log_dir=model.log_dir, update_freq=model.save_freq,
                                     wrapper_model=model))

    model.fit(dataset_train, validation_dataset=dataset_val, epochs=EPOCHS, callbacks=callbacks)
    model.model.summary()
