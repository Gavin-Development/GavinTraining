from GavinBackend.utils.model_management import *


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Set to no logs output.


# noinspection PyBroadException
def test_model_load():
    hparams = {"MAX_SAMPLES": 1000000,
               "MODEL_NAME": "Test",
               "BATCH_SIZE": 64,
               "BUFFER_SIZE": 20000,
               "NUM_LAYERS": 4,
               "D_MODEL": 256,
               "NUM_HEADS": 8,
               "UNITS": 4096,
               "DROPOUT": 0.1,
               "VOCAB_SIZE": 8096,
               "EPOCHS": 20,
               "OTHER_POLICY": False}
    try:
        model, dataset_t, dataset_v = create_model("models", "D:\\Datasets\\reddit_data\\files\\",
                                                   hparams=hparams)
    except Exception as e:
        assert False, f"Error: {e}"
