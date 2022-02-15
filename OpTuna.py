from dotenv import load_dotenv

load_dotenv()

import os
import platform
import optuna

os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

from GavinBackend.GavinCore.models import TransformerIntegration, tf, tfds, PerformerIntegration, FNetIntegration
from GavinBackend.GavinCore.datasets import DatasetAPICreator
from GavinBackend.DataParsers.load_data import load_tokenized_data
from GavinBackend.GavinCore.metrics import Perplexity

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
if not os.path.exists("bunchOfLogs/optuna"):
    os.mkdir("./bunchOfLogs/optuna/")


DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
STUDY_NAME = os.getenv("STUDY_NAME")
print("DB_HOST:", DB_HOST)
print("DB_PORT:", DB_PORT)
print("DB_USER:", DB_USER)
print("DB_PASSWORD:", DB_PASSWORD)
print("DB_NAME:", DB_NAME)
print("STUDY_NAME:", STUDY_NAME)

if None in [DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, STUDY_NAME]:
    raise ValueError("Please set DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME")

PYTHON_LEGACY = False if "windows" in platform.system().lower() else True
CPP_LEGACY = False
DATASET_PATH = input("Please enter dataset path: ")
MAX_SAMPLES = 100_000
BATCH_SIZE = 16
BUFFER_SIZE = 5_000
TOKENIZER_PATH = "Tokenizer-3"
dataset_file_name = "Tokenizer-3"
EPOCHS = 5
tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(TOKENIZER_PATH)
metadata = {'MAX_SAMPLES': MAX_SAMPLES, 'BATCH_SIZE': BATCH_SIZE, 'BUFFER_SIZE': BUFFER_SIZE}
SAVE_FREQ = 500
kwargs = {'save_freq': SAVE_FREQ, 'batch_size': BATCH_SIZE, 'tokenizer': tokenizer,
          'base_log_dir': "bunchOfLogs/optuna"}


def create_model(trail: optuna.trial.Trial):
    tf.keras.backend.clear_session()
    kwargs['name'] = f"{trail.number}-Gavin-Optuna"
    model_options = ["Transformer", "Performer", "FNet"]
    model_option = trail.suggest_categorical("model_option", model_options)
    if model_option == "Transformer":
        model_type = TransformerIntegration
    elif model_option == "Performer":
        model_type = PerformerIntegration
    elif model_option == "FNet":
        model_type = FNetIntegration
    else:
        raise ValueError(f"Unknown Model Option: {model_option}")
    max_length = trail.suggest_int("MAX_LENGTH", 10, 100, step=10)
    num_layers = trail.suggest_int("NUM_LAYERS", 2, 10, step=1)
    d_model = trail.suggest_int("D_MODEL", 128, 2048, step=64)
    num_heads = trail.suggest_int("NUM_HEADS", 2, 16, step=2)
    units = trail.suggest_int("UNITS", 512, 4096, step=512)
    dropout = trail.suggest_float("DROPOUT", 0.01, 0.1, step=0.02)
    kwargs['max_len'] = max_length
    kwargs['num_layers'] = num_layers
    kwargs['d_model'] = d_model
    kwargs['num_heads'] = num_heads
    kwargs['units'] = units
    kwargs['dropout'] = dropout
    if model_option == "Performer":
        num_features = trail.suggest_int("num_features", d_model // 2, d_model, step=d_model // 4)
        kwargs['num_features'] = num_features
    print(f"Model Type: {model_option}")
    options_string = ""
    for k, v in kwargs.items():
        options_string += f"{k}={v}\n"
    print(f"Model Options: \n\n{options_string}")
    return model_type(**kwargs)


def objective(trial: optuna.trial.Trial):
    model = create_model(trial)
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
    history = model.fit(dataset_train, validation_dataset=dataset_val, epochs=EPOCHS, callbacks=callbacks)
    return history.history['sparse_categorical_accuracy'][-1]


if __name__ == "__main__":
    # study = optuna.create_study(direction="maximize")
    study = optuna.load_study(
        study_name=STUDY_NAME,
        storage=f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
    )
    study.optimize(objective, n_trials=100, catch=(tf.errors.ResourceExhaustedError, AssertionError))

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    best_trial = study.best_trial

    print("  Value: ", best_trial.value)

    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))
