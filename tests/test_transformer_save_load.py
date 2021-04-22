from pathlib import Path
from GavinBackend.utils.model_management import *


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Set to no logs output.
BASE_DIR = Path(__file__).resolve().parent.parent


# noinspection PyBroadException
def test_new_model_load():
    hparams = {"MAX_SAMPLES": 100000,
               "MODEL_NAME": "Test",
               "MAX_LENGTH": 42,
               "BATCH_SIZE": 64,
               "BUFFER_SIZE": 20000,
               "NUM_LAYERS": 4,
               "D_MODEL": 256,
               "NUM_HEADS": 8,
               "UNITS": 4096,
               "DROPOUT": 0.1,
               "VOCAB_SIZE": 8096,
               "EPOCHS": 20,
               "OTHER_POLICY": False,
               "TOKENIZER_NAME": "Tokenizer-3"}
    try:
        model, dataset_t, dataset_v, _ = create_model("models", "D:\\Datasets\\reddit_data\\files\\",
                                                      hparams=hparams, tokenizer_path=
                                                      os.path.join(os.path.join(BASE_DIR, "tests/other_files/"),
                                                                   "Tokenizer-3"))
    except tf.errors.ResourceExhaustedError as e:
        assert False, f"Resources Run Out: {e}"
    except Exception as e:
        assert False, f"Error: {e}"
    finally:
        del model, dataset_t, dataset_v


# noinspection PyBroadException
def test_old_model_load():
    try:
        model, dataset_t, dataset_v, hparams = model_load("models", "D:\\Datasets\\reddit_data\\files\\", "Gerald_V3")
    except Exception as e:
        assert False, f"Error: {e}"
    except tf.errors.ResourceExhaustedError as e:
        assert False, f"Resources Run Out: {e}"
    finally:
        del model, dataset_t, dataset_v, hparams
