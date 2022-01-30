if __name__ == "__main__":
    import os
    import uuid
    import datetime
    import time
    import shutil
    import logging
    from itertools import product

    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    import tensorflow as tf
    import tensorflow_datasets as tfds
    from tensorboard.plugins.hparams import api as hp
    from GavinBackend.GavinCore.models import TransformerIntegration, PerformerIntegration
    from GavinBackend.GavinCore.datasets import DatasetAPICreator
    from GavinBackend.DataParsers.load_data import load_tokenized_data

    isPreformer = True
    logging.basicConfig(level=logging.INFO,
                        format='%(process)d-%(levelname)s %(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger(__name__)
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except Exception as e:
        logger.warning(f"Error on Memory Growth Setting. {e}")
    else:
        logger.info("Memory Growth Set to True.")

    HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([1024]))
    HP_MAX_SAMPLES = hp.HParam('max_samples', hp.Discrete([1_000_000, 2_000_000, 5_000_000]))
    HP_D_MODEL = hp.HParam('d_model', hp.Discrete([256]))
    HP_NUM_FEATURES = hp.HParam('num_features', hp.Discrete([128, 256]))
    HP_NUM_LAYERS = hp.HParam('num_layers', hp.Discrete([2, 4]))

    # Non-testing params defaults
    HP_NUM_HEADS = hp.HParam('num_heads', hp.Discrete([8]))
    HP_MAX_LENGTH = hp.HParam('max_length', hp.Discrete([30]))
    HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.1))
    HP_BUFFER_SIZE = hp.HParam('buffer_size', hp.Discrete([20_000]))
    HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([32]))
    HP_WARMUP_STEPS = hp.HParam('warmup_steps', hp.Discrete([4000]))
    HP_EPOCHS = hp.HParam('epochs', hp.Discrete([3]))

    test_log_dir_root = './bunchOfLogs/hparams/'
    if not os.path.exists(test_log_dir_root):
        os.mkdir(test_log_dir_root)
    tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file('Tokenizer-3')
    session_num = 0
    hparams = [HP_NUM_UNITS, HP_MAX_SAMPLES, HP_D_MODEL, HP_NUM_FEATURES, HP_NUM_LAYERS, HP_NUM_HEADS,
               HP_MAX_LENGTH, HP_DROPOUT, HP_BUFFER_SIZE, HP_BATCH_SIZE, HP_WARMUP_STEPS, HP_EPOCHS]
    num_runs = 1
    for hparam in hparams:
        if type(hparam) == hp.Discrete:
            num_runs *= len(hparam.domain.values)
        elif type(hparam) == hp.RealInterval:
            num_runs *= (hparam.domain.max_value - hparam.domain.min_value)
    product_args = (HP_NUM_UNITS.domain.values, HP_MAX_SAMPLES.domain.values, HP_D_MODEL.domain.values, HP_NUM_FEATURES.domain.values, HP_NUM_LAYERS.domain.values, HP_NUM_HEADS.domain.values,
                    HP_MAX_LENGTH.domain.values, (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value), HP_BUFFER_SIZE.domain.values,
                    HP_BATCH_SIZE.domain.values, HP_WARMUP_STEPS.domain.values, HP_EPOCHS.domain.values)
    for num_units, max_samples, d_model, num_features, num_layers, num_heads, max_length, dropout, buffer_size, batch_size, warmup_steps, epochs in product(
            *product_args):
        tf.keras.backend.clear_session()  # Reduces the amount of memory this will use.

        run_name = "run-%d" % session_num
        logger.info(
            f"--- Starting trial: {run_name} Completed: {round(((session_num / num_runs) * 100), 2)}%")
        logger.info(f"""\
NUM_LAYERS: {num_layers}
NUM_UNITS: {num_units}
D_MODEL: {d_model}
NUM_FEATURES: {num_features if isPreformer else None}
MAX_SAMPLES: {max_samples}
NUM_HEADS: {num_heads}
DROPOUT: {dropout}
MAX_LENGTH: {max_length}
BUFFER_SIZE: {buffer_size}
BATCH_SIZE: {batch_size}
WARMUP_STEPS: {warmup_steps}
EPOCHS: {epochs}
""")
        LOG_DIR = test_log_dir_root
        MODEL_NAME = f'{str(uuid.uuid4()).upper()}--{datetime.datetime.now().strftime("%m-%d-%Y")}'
        try:
            if not isPreformer:
                model = TransformerIntegration(num_layers=num_layers,
                                               units=num_units,
                                               d_model=d_model,
                                               num_heads=num_heads,
                                               base_log_dir=LOG_DIR,
                                               dropout=dropout,
                                               max_len=max_length,
                                               tokenizer=tokenizer,
                                               name=MODEL_NAME,
                                               batch_size=batch_size)
            else:
                model = PerformerIntegration(num_layers=num_layers,
                                             units=num_units,
                                             d_model=d_model,
                                             num_features=num_features,
                                             num_heads=num_heads,
                                             base_log_dir=LOG_DIR,
                                             dropout=dropout,
                                             max_len=max_length,
                                             tokenizer=tokenizer,
                                             name=MODEL_NAME,
                                             batch_size=batch_size)
            questions, answers = load_tokenized_data(
                max_samples=max_samples,
                data_path="D:\\Datasets\\reddit_data\\files\\",
                filename='Tokenizer-3',
                s_token=model.start_token,
                e_token=model.end_token, max_len=max_length)
            # questions = tf.keras.preprocessing.sequence.pad_sequences(questions,
            # maxlen=model.max_len,
            # padding='post')
            # answers = tf.keras.preprocessing.sequence.pad_sequences(answers,
            # maxlen=model.max_len,
            # padding='post')
            dataset_train, dataset_val = DatasetAPICreator.create_data_objects(
                questions, answers,
                buffer_size=buffer_size,
                batch_size=batch_size, vocab_size=model.vocab_size)
            del questions, answers
            callbacks = model.get_default_callbacks()[
                        :-1]  # Remove the predict callback
            hparams = {HP_NUM_LAYERS: num_layers,
                       HP_NUM_UNITS: num_units,
                       HP_D_MODEL: d_model,
                       HP_MAX_SAMPLES: max_samples,
                       HP_NUM_HEADS: num_heads,
                       HP_MAX_LENGTH: max_length,
                       HP_DROPOUT: dropout,
                       HP_BUFFER_SIZE: buffer_size,
                       HP_BATCH_SIZE: batch_size,
                       HP_WARMUP_STEPS: warmup_steps,
                       HP_EPOCHS: epochs}
            if isPreformer:
                hparams[HP_NUM_FEATURES] = num_features
            callbacks.insert(0,
                             hp.KerasCallback(model.log_dir,
                                              hparams))
            del callbacks[1]  # Remove Tensorboard
            # callbacks.append(
            #   tf.keras.callbacks.TensorBoard(log_dir=model.log_dir))
            model.fit(dataset_train, validation_dataset=dataset_val,
                      epochs=epochs,
                      callbacks=callbacks)
            session_num += 1
        except Exception as e:
            logger.error(
                f"""[{datetime.datetime.now().strftime('%m-%d-%Y %H:%M:%S.%f')[:-2]}] Logged Error on Run: {run_name}
Model: {MODEL_NAME} 
Stats: 
-NUM_LAYERS: {num_layers}
-NUM_UNITS: {num_units}
-D_MODEL: {d_model}
-MAX_SAMPLES: {max_samples}
-NUM_HEADS: {num_heads}
-DROPOUT: {dropout}
-BUFFER_SIZE: {buffer_size}
-BATCH_SIZE: {batch_size}
-WARMUP_STEPS: {warmup_steps}
-EPOCHS: {epochs}
Error: {e}\n\n""")
            shutil.rmtree(os.path.join(LOG_DIR, MODEL_NAME))
            time.sleep(2)
            continue
    logger.info(
        f"[{datetime.datetime.now().strftime('%m-%d-%Y %H:%M:%S.%f')[:-2]}] Finished all runs.")
