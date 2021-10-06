import os
import json
import platform

os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
if __name__ == "__main__":
    from GavinBackend.GavinCore.models import TransformerIntegration, tf, tfds, PerformerIntegration
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

    LEGACY = False if "windows" in platform.system().lower() else True
    DATASET_PATH = input("Please enter dataset path: ")
    MODEL_TYPE = "performer"
    if MODEL_TYPE.lower() == "performer":
        MODEL_TYPE = PerformerIntegration
    elif MODEL_TYPE.lower() == "transformer":
        MODEL_TYPE = TransformerIntegration
    else:
        print("Invalid model type. Quitting")
        quit()
    LOG_DIR = './bunchOfLogs'
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)
    MODEL_NAME = "ShmarvRun"
    if os.path.exists(os.path.join(LOG_DIR, MODEL_NAME)):
        metadata = json.load(
            open(os.path.join(LOG_DIR, os.path.join(MODEL_NAME, os.path.join('config/', 'metadata.json')))))
        choice = "n"
        if choice == "y":
            MAX_SAMPLES = 80_000_000
            BATCH_SIZE = 128
            BUFFER_SIZE = 50_000
        else:
            MAX_SAMPLES = metadata['MAX_SAMPLES']
            BATCH_SIZE = metadata['BATCH_SIZE']
            BUFFER_SIZE = metadata['BUFFER_SIZE']
    else:
        MAX_SAMPLES = 80_000_000
        BATCH_SIZE = 128
        BUFFER_SIZE = 50_000
    TOKENIZER_PATH = "Tokenizer-3"
    EPOCHS = 50
    tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(TOKENIZER_PATH)
    if os.path.exists(os.path.join(LOG_DIR, MODEL_NAME)):
        model = MODEL_TYPE.load_model(LOG_DIR, MODEL_NAME)
        model.metadata = metadata
        questions, answers = load_tokenized_data(max_samples=MAX_SAMPLES,
                                                 data_path=DATASET_PATH,
                                                 tokenizer_name=os.path.basename(TOKENIZER_PATH),
                                                 s_token=model.start_token,
                                                 e_token=model.end_token, max_len=model.max_len, legacy=LEGACY)
        if LEGACY:
            questions = tf.keras.preprocessing.sequence.pad_sequences(questions, maxlen=model.max_len, padding='post')
            answers = tf.keras.preprocessing.sequence.pad_sequences(answers, maxlen=model.max_len, padding='post')
        dataset_train, dataset_val = DatasetAPICreator.create_data_objects(questions, answers, buffer_size=BUFFER_SIZE,
                                                                           batch_size=BATCH_SIZE,
                                                                           vocab_size=model.vocab_size)
        callbacks = model.get_default_callbacks()
        del callbacks[1]  # Have to remove tensorboard, due to access violation errors
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=model.log_dir))  # Add tensorboard without profile
        model.fit(dataset_train, validation_dataset=dataset_val, epochs=EPOCHS, callbacks=callbacks)
    else:
        MAX_LENGTH = 100
        NUM_LAYERS = 12
        D_MODEL = 512
        NUM_HEADS = 16
        UNITS = 4096
        DROPOUT = 0.2
        SAVE_FREQ = input("Press Enter to save by epoch, or type a number to save by batch: ")
        if SAVE_FREQ == "\n" or SAVE_FREQ == "":
            SAVE_FREQ = 'epoch'
        else:
            SAVE_FREQ = int(SAVE_FREQ)

        metadata = {"MAX_SAMPLES": MAX_SAMPLES, "BATCH_SIZE": BATCH_SIZE, "BUFFER_SIZE": BUFFER_SIZE}
        if MODEL_TYPE == TransformerIntegration:
            model = MODEL_TYPE(num_layers=NUM_LAYERS, units=UNITS, d_model=D_MODEL,
                               num_heads=NUM_HEADS, base_log_dir=LOG_DIR, dropout=DROPOUT,
                               max_len=MAX_LENGTH, tokenizer=tokenizer, name=MODEL_NAME,
                               metadata=metadata,
                               metrics=['accuracy', Perplexity(max_len=MAX_LENGTH, vocab_size=tokenizer.vocab_size)],
                               save_freq=SAVE_FREQ, batch_size=BATCH_SIZE)
        else:
            NUM_FEATURES = int(input("RANDOM_FEATURES: "))
            model = MODEL_TYPE(num_layers=NUM_LAYERS, units=UNITS, d_model=D_MODEL,
                               num_heads=NUM_HEADS, base_log_dir=LOG_DIR, dropout=DROPOUT,
                               max_len=MAX_LENGTH, tokenizer=tokenizer, name=MODEL_NAME,
                               metadata=metadata, num_features=NUM_FEATURES,
                               metrics=['accuracy', Perplexity(max_len=MAX_LENGTH, vocab_size=tokenizer.vocab_size)],
                               save_freq=SAVE_FREQ, batch_size=BATCH_SIZE)
        questions, answers = load_tokenized_data(max_samples=MAX_SAMPLES,
                                                 data_path=DATASET_PATH,
                                                 tokenizer_name=os.path.basename(TOKENIZER_PATH),
                                                 s_token=model.start_token,
                                                 e_token=model.end_token, max_len=MAX_LENGTH, legacy=LEGACY)
        if LEGACY:
            questions = tf.keras.preprocessing.sequence.pad_sequences(questions, maxlen=model.max_len, padding='post')
            answers = tf.keras.preprocessing.sequence.pad_sequences(answers, maxlen=model.max_len, padding='post')
        dataset_train, dataset_val = DatasetAPICreator.create_data_objects(questions, answers, buffer_size=BUFFER_SIZE,
                                                                           batch_size=BATCH_SIZE, vocab_size=model.vocab_size)

        callbacks = model.get_default_callbacks()
        del callbacks[1]  # Have to remove tensorboard, due to access violation errors
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=model.log_dir))  # Add tensorboard without profile
        model.fit(dataset_train, validation_dataset=dataset_val, epochs=EPOCHS, callbacks=callbacks)
        model.model.summary()
