SAVE_FREQ = input("Press Enter to save by epoch, or type a number to save by batch: ")
if SAVE_FREQ == "\n":
    SAVE_FREQ = 'epoch'
else:
    SAVE_FREQ = int(SAVE_FREQ)
import os
import json

os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
if __name__ == "__main__":
    from GavinBackend.GavinCore.models import TransformerIntegration, tf, tfds, PreformerIntegration
    from GavinBackend.GavinCore.datasets import create_data_objects
    from GavinBackend.DataParsers.load_data import load_tokenized_data

    MODEL_TYPE = "transformer"  # input("Please enter a Model Type [`preformer`, `transformer`]: ")
    if MODEL_TYPE.lower() == "performer":
        MODEL_TYPE = PreformerIntegration
    elif MODEL_TYPE.lower() == "transformer":
        MODEL_TYPE = TransformerIntegration
    else:
        print("Invalid model type. Quitting")
        quit()
    LOG_DIR = './bunchOfLogs'
    MODEL_NAME = "Gavin_Big"  # input("Please enter Model_Name: ")
    if os.path.exists(os.path.join(LOG_DIR, MODEL_NAME)):
        metadata = json.load(
            open(os.path.join(LOG_DIR, os.path.join(MODEL_NAME, os.path.join('config/', 'metadata.json')))))
        MAX_SAMPLES = metadata['MAX_SAMPLES']
        BATCH_SIZE = metadata['BATCH_SIZE']
        BUFFER_SIZE = metadata['BUFFER_SIZE']
    else:
        MAX_SAMPLES = 85_000_000  # int(input("MAX_SAMPLES: "))
        BATCH_SIZE = 256  # int(input("BATCH_SIZE: "))
        BUFFER_SIZE = 20_0000  # int(input("BUFFER_SIZE: "))
    TOKENIZER_PATH = "Tokenizer-3"  # input("TOKENIZER_PATH: ")
    EPOCHS = 30  # int(input("EPOCHS: "))
    tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(TOKENIZER_PATH)
    if os.path.exists(os.path.join(LOG_DIR, MODEL_NAME)):
        model = MODEL_TYPE.load_model(LOG_DIR, MODEL_NAME)
        model.metadata = metadata
        questions, answers = load_tokenized_data(max_samples=MAX_SAMPLES,
                                                 data_path=".",
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
    else:
        MAX_LENGTH = 80  # int(input("MAX_LENGTH: "))
        NUM_LAYERS = 8  # int(input("NUM_LAYERS: "))
        D_MODEL = 512  # int(input("D_MODEL: "))
        NUM_HEADS = 32  # int(input("NUM_HEADS: "))
        UNITS = 8192  # int(input("UNITS: "))
        DROPOUT = 0.15  # float(input("DROPOUT: "))
        metadata = {"MAX_SAMPLES": MAX_SAMPLES, "BATCH_SIZE": BATCH_SIZE, "BUFFER_SIZE": BUFFER_SIZE}
        if MODEL_TYPE == TransformerIntegration:
            model = MODEL_TYPE(num_layers=NUM_LAYERS, units=UNITS, d_model=D_MODEL,
                               num_heads=NUM_HEADS, base_log_dir=LOG_DIR, dropout=DROPOUT,
                               max_len=MAX_LENGTH, tokenizer=tokenizer, name=MODEL_NAME,
                               metadata=metadata, save_freq=SAVE_FREQ)
        else:
            NUM_FEATURES = 128
            model = MODEL_TYPE(num_layers=NUM_LAYERS, units=UNITS, d_model=D_MODEL,
                               num_heads=NUM_HEADS, base_log_dir=LOG_DIR, dropout=DROPOUT,
                               max_len=MAX_LENGTH, tokenizer=tokenizer, name=MODEL_NAME,
                               metadata=metadata, num_features=NUM_FEATURES, save_freq=SAVE_FREQ)
        questions, answers = load_tokenized_data(max_samples=MAX_SAMPLES,
                                                 data_path=".",
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
