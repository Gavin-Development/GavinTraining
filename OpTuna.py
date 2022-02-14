import os
import json
import platform
import shutil
import optuna

os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

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

PYTHON_LEGACY = False if "windows" in platform.system().lower() else True
CPP_LEGACY = False
DATASET_PATH = input("Please enter dataset path: ")
MAX_SAMPLES = 5_000_000
BATCH_SIZE = 32
BUFFER_SIZE = 5_000
TOKENIZER_PATH = "Tokenizer-3"
dataset_file_name = "Tokenizer-3"
EPOCHS = 10
tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(TOKENIZER_PATH)
metadata = {'MAX_SAMPLES': MAX_SAMPLES, 'BATCH_SIZE': BATCH_SIZE, 'BUFFER_SIZE': BUFFER_SIZE}
SAVE_FREQ = 2000
kwargs = {'save_freq': SAVE_FREQ, 'batch_size': BATCH_SIZE, 'tokenizer': tokenizer,
          'base_log_dir': "bunchOfLogs/optuna"}


def create_model(trail: optuna.trial.Trial):
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
    max_length = trail.suggest_int("MAX_LENGTH", 10, 100)
    num_layers = trail.suggest_int("NUM_LAYERS", 1, 10)
    d_model = trail.suggest_int("D_MODEL", 128, 512)
    num_heads = trail.suggest_int("NUM_HEADS", 1, 8)
    units = trail.suggest_int("UNITS", 512, 4096)
    dropout = trail.suggest_float("DROPOUT", 0.1, 0.5)
    kwargs['max_len'] = max_length
    kwargs['num_layers'] = num_layers
    kwargs['d_model'] = d_model
    kwargs['num_heads'] = num_heads
    kwargs['units'] = units
    kwargs['dropout'] = dropout
    if model_option == "Performer":
        num_features = trail.suggest_int("num_features", d_model // 2, d_model)
        kwargs['num_features'] = num_features

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
    return history.history['accuracy'][-1]


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
