import os
import json

os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
if __name__ == "__main__":
    from GavinBackend.GavinCore.models import TransformerIntegration, tf, tfds, PerformerIntegration
    from GavinBackend.GavinCore.datasets import create_data_objects
    from GavinBackend.DataParsers.load_data import load_tokenized_data

    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except Exception as e:
        print(f"Error on Memory Growth Setting. {e}")
    else:
        print("Memory Growth Set to True.")

    MODEL_TYPE = input("Please enter a Model Type [`performer`, `transformer`]: ")
    if MODEL_TYPE.lower() == "performer":
        MODEL_TYPE = PerformerIntegration
    elif MODEL_TYPE.lower() == "transformer":
        MODEL_TYPE = TransformerIntegration
    else:
        print("Invalid model type. Quitting")
        quit()
    LOG_DIR = './bunchOfLogs'
    MODEL_NAME = input("Please enter Model_Name: ")
    if os.path.exists(os.path.join(LOG_DIR, MODEL_NAME)):
        metadata = json.load(
            open(os.path.join(LOG_DIR, os.path.join(MODEL_NAME, os.path.join('config/', 'metadata.json')))))
        MAX_SAMPLES = metadata['MAX_SAMPLES']
        BATCH_SIZE = metadata['BATCH_SIZE']
        BUFFER_SIZE = metadata['BUFFER_SIZE']
    else:
        MAX_SAMPLES = int(input("MAX_SAMPLES: "))
        BATCH_SIZE = int(input("BATCH_SIZE: "))
        BUFFER_SIZE = int(input("BUFFER_SIZE: "))
    TOKENIZER_PATH = input("TOKENIZER_PATH: ")
    EPOCHS = int(input("EPOCHS: "))
    tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(TOKENIZER_PATH)
    if os.path.exists(os.path.join(LOG_DIR, MODEL_NAME)):
        model = MODEL_TYPE.load_model(LOG_DIR, MODEL_NAME)
        model.metadata = metadata
        questions, answers = load_tokenized_data(max_samples=MAX_SAMPLES,
                                                 data_path="D:\\Datasets\\reddit_data\\files\\",
                                                 tokenizer_name=os.path.basename(TOKENIZER_PATH),
                                                 s_token=model.start_token,
                                                 e_token=model.end_token,)
        questions = tf.keras.preprocessing.sequence.pad_sequences(questions, maxlen=model.max_len, padding='post')
        answers = tf.keras.preprocessing.sequence.pad_sequences(answers, maxlen=model.max_len, padding='post')
        dataset_train, dataset_val = create_data_objects(questions, answers, buffer_size=BUFFER_SIZE,
                                                         batch_size=BATCH_SIZE)
        callbacks = model.get_default_callbacks()
        del callbacks[1]  # Have to remove tensorboard, due to access violation errors
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=model.log_dir))  # Add tensorboard without profile
        model.fit(dataset_train, validation_dataset=dataset_val, epochs=EPOCHS, callbacks=callbacks)
    else:
        MAX_LENGTH = int(input("MAX_LENGTH: "))
        NUM_LAYERS = int(input("NUM_LAYERS: "))
        D_MODEL = int(input("D_MODEL: "))
        NUM_HEADS = int(input("NUM_HEADS: "))
        UNITS = int(input("UNITS: "))
        DROPOUT = float(input("DROPOUT: "))
        metadata = {"MAX_SAMPLES": MAX_SAMPLES, "BATCH_SIZE": BATCH_SIZE, "BUFFER_SIZE": BUFFER_SIZE}
        if MODEL_TYPE == TransformerIntegration:
            model = MODEL_TYPE(num_layers=NUM_LAYERS, units=UNITS, d_model=D_MODEL,
                               num_heads=NUM_HEADS, base_log_dir=LOG_DIR, dropout=DROPOUT,
                               max_len=MAX_LENGTH, tokenizer=tokenizer, name=MODEL_NAME,
                               metadata=metadata, metrics=['accuracy'])
        else:
            NUM_FEATURES = int(input("RANDOM_FEATURES: "))
            model = MODEL_TYPE(num_layers=NUM_LAYERS, units=UNITS, d_model=D_MODEL,
                               num_heads=NUM_HEADS, base_log_dir=LOG_DIR, dropout=DROPOUT,
                               max_len=MAX_LENGTH, tokenizer=tokenizer, name=MODEL_NAME,
                               metadata=metadata, num_features=NUM_FEATURES, metrics=['accuracy'])
        questions, answers = load_tokenized_data(max_samples=MAX_SAMPLES,
                                                 data_path="D:\\Datasets\\reddit_data\\files\\",
                                                 tokenizer_name=os.path.basename(TOKENIZER_PATH),
                                                 s_token=model.start_token,
                                                 e_token=model.end_token, )
        questions = tf.keras.preprocessing.sequence.pad_sequences(questions, maxlen=model.max_len, padding='post')
        answers = tf.keras.preprocessing.sequence.pad_sequences(answers, maxlen=model.max_len, padding='post')
        dataset_train, dataset_val = create_data_objects(questions, answers, buffer_size=BUFFER_SIZE,
                                                         batch_size=BATCH_SIZE)

        callbacks = model.get_default_callbacks()
        del callbacks[1]  # Have to remove tensorboard, due to access violation errors
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=model.log_dir))  # Add tensorboard without profile
        model.fit(dataset_train, validation_dataset=dataset_val, epochs=EPOCHS, callbacks=callbacks)
