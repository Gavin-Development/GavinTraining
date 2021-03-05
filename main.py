if __name__ == "__main__":
    import os
    os.environ['TF_GPU_THREAD_MODE'] = "gpu_private"
    os.environ['REDDIT_DATASET_PATH'] = "D:\\Datasets\\reddit_data\\files\\"
    DATASET_PATH = os.getenv('REDDIT_DATASET_PATH')
    import numpy as np
    from datetime import datetime
    # import tensorflow as tf Not needed since its imported through GavinBackend.models
    import tensorflow_datasets as tfds
    import GavinBackend.preprocessing.text as gbpte
    import GavinBackend.preprocessing.concurrent as gbpc
    import GavinBackend.preprocessing.tokenise as gbpt
    import GavinBackend.functions as gbf

    from tensorboard.plugins import projector
    from GavinBackend.models import Transformer, tf, DocumentLevelContextTransformer
    from GavinBackend.callbacks.model_callbacks import PredictCallback
    from tensorflow.keras.utils import plot_model
    from tensorflow.keras.mixed_precision import experimental as mixed_precision

    other_policy = 'n'  # input("Do you want to enabled mixed precision? y/n (NOT SUPPORTED YET): ")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if other_policy == 'y':
        MIXED = True
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(f"{len(gpus)} Physical GPUS, {len(logical_gpus)} Logical GPUS.")
            except RuntimeError as e:
                print(e)
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)
    else:
        MIXED = False
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.compat.v1.Session(config=config)

    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Numpy Version: {np.__version__}")
    print(f"Eager execution: {tf.executing_eagerly()}")

    path_to_dataset = "cornell movie-dialogs corpus"

    path_to_movie_lines = os.path.join(path_to_dataset, "movie_lines.txt")
    path_to_movie_conversations = os.path.join(path_to_dataset, "movie_conversations.txt")

    # User Input Data
    MAX_SAMPLES = int(input("MAX_SAMPLES: "))
    name = input("Please enter a ModelName for this train: ")
    log_dir = "bunchOfLogs/" + name
    BATCH_SIZE = int(input("BATCH_SIZE(32): "))
    BUFFER_SIZE = 40_000
    MAX_LENGTH = 50 + 2

    # Hyper-parameters
    NUM_LAYERS = int(input("Please enter the number of NUM_LAYERS(4): "))
    D_MODEL = int(input("Please enter the d_model(256): "))
    NUM_HEADS = int(input("Please enter the NUM_HEADS(8): "))
    UNITS = int(input("Please enter the number of units(512): "))
    DROPOUT = float(input("Please enter the DROPOUT(0.175): "))
    EPOCHS = int(input("Please enter the number of epochs(15): "))
    load = input("Would you like to load the tokenizer? y/n: ")
    tokenizerPath = None
    if load == "y":
        tokenizerPath = input("Please enter the path the tokenizer: ")
    cores = int(input("How many cores would you like to use for pre-processing: "))
    regex_cores = cores
    TARGET_VOCAB_SIZE = 2 ** 14

    checkpoint_path = f"{log_dir}/cp.ckpt"
    try:
        os.mkdir(f"{log_dir}")
        os.mkdir(f"{log_dir}/model/")
        os.mkdir(f"{log_dir}/pickles/")
        os.mkdir(f"{log_dir}/tokenizer")
        os.mkdir(f"{log_dir}/values/")
        os.mkdir(f"{log_dir}/images/")
        os.mkdir(f"{log_dir}/logs/")
    except FileExistsError:
        print("Already exists not creating folders")
        pass

    print("Loading files...")
    questions, answers = gbpte.load_data(MAX_SAMPLES, DATASET_PATH)
    print("Done loading...")

    if load == "n":  # If we're not loading the tokenizer then generate this
        print("Starting Tokenizer this may take a while....")
        # Build tokenizer using tfds for both questions and answers
        tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            questions + answers, target_vocab_size=TARGET_VOCAB_SIZE)
        tokenizer.save_to_file(f"{log_dir}/tokenizer/vocabTokenizer")
    else:  # load the tokenizer
        tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(tokenizerPath)
        tokenizer.save_to_file(f"{log_dir}/tokenizer/vocabTokenizer")
    print("Done Tokenizer.")

    # Define start and end token to indicate the start and end of a sentence
    START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]  # Set the START and END tokens

    # Vocabulary size plus start and end token
    VOCAB_SIZE = tokenizer.vocab_size + 2  # In create the vocab size to account for the start end token

    print(f"Pickling Questions and answers for {name}")
    questionsMarshal = f"{log_dir}/pickles/{name}_questions.marshal"
    answersMarshal = f"{log_dir}/pickles/{name}_answers.marshal"
    # gbpc.save_files(questions, answers, questionsMarshal, answersMarshal)
    print(f"Done saving....")
    mirrored_strategy = tf.distribute.MirroredStrategy()  # Use mirrored strategy to use multi gpu
    print("Filtering data")
    questions, answers = gbpt.tokenize_and_filter(questions, answers, cores, MAX_LENGTH, START_TOKEN, END_TOKEN, tokenizer)  # Filter all the data
    sizes = (len(questions), len(answers))
    print(f"Answers: {sizes[1]}\nQuestions: {sizes[0]}")
    questions_train = questions[0: int(sizes[0] * .80)]
    questions_val = questions[int(sizes[0] * 0.80):]
    answers_train = answers[0: int(sizes[1] * .80)]
    answers_val = answers[int(sizes[1] * .80):]
    print("Done filtering")
    # decoder inputs use the previous target as input
    # remove s_token from targets
    # print("Beginning Dataset Shuffling, Batching and Prefetch.")
    dataset_train = tf.data.Dataset.from_tensor_slices((
        {
            'inputs': questions_train,  # Source
            'dec_inputs': answers_train[:, :-1]  # Targets
        },
        {
            'outputs': answers_train[:, 1:]  # Outputs
        }))
    dataset_val = tf.data.Dataset.from_tensor_slices((
        {
            'inputs': questions_val,  # Source
            'dec_inputs': answers_val[:, :-1]  # Targets
            },
        {
            'outputs': answers_val[:, 1:]  # Outputs
        }))

    dataset_train = dataset_train.cache()
    dataset_val = dataset_val.cache()
    dataset_train = dataset_train.shuffle(BUFFER_SIZE)
    dataset_val = dataset_val.shuffle(BUFFER_SIZE)
    dataset_train = dataset_train.batch(BATCH_SIZE)
    dataset_val = dataset_val.batch(BATCH_SIZE)
    dataset_train = dataset_train.prefetch(tf.data.experimental.AUTOTUNE)
    dataset_val = dataset_val.prefetch(tf.data.experimental.AUTOTUNE)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    dataset_train.with_options(options)
    dataset_val.with_options(options)
    print("Done Dataset shuffling, batching and prefetch")
    with mirrored_strategy.scope():  # Use the mirrored strategy to create the model
        transformer = Transformer(
            vocab_size=VOCAB_SIZE,
            num_layers=NUM_LAYERS,
            units=UNITS,
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            dropout=DROPOUT,
            mixed=MIXED)
        model = transformer.return_model()

    # noinspection PyAbstractClass,PyShadowingNames
    class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

        def __init__(self, d_model, warmup_steps=5000):
            super(CustomSchedule, self).__init__()

            self.d_model = d_model
            self.d_model = tf.cast(self.d_model, tf.float32)

            self.warmup_steps = warmup_steps

        def __call__(self, step):
            step = tf.cast(step, tf.float32)
            arg1 = tf.math.rsqrt(step)
            arg2 = step * (self.warmup_steps ** -1.5)

            return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

        def get_config(self):
            config = {
                'd_model': self.d_model,
                'warmup_steps': self.warmup_steps
            }
            return config


    learning_rate = CustomSchedule(D_MODEL)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.91, beta_2=0.98, epsilon=1e-9)

    print("Writing metadata")

    with open(os.path.join(log_dir, 'metadata.tsv'), "w", encoding="utf-8") as f:
        for subwords in tokenizer.subwords:
            f.write(f"{subwords}\n")
        for unknown in range(1, tokenizer.vocab_size - len(tokenizer.subwords)):
            f.write(f"unknown #{unknown}\n")

    projector_config = projector.ProjectorConfig()
    embedding = projector_config.embeddings.add()

    embedding.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(log_dir, projector_config)

    linebreak = "--------------------------------"
    log = f"""\nDate: {datetime.now().strftime("%d/%m/%Y %H-%M-%S")},
     Name: {name},
     PATH: {checkpoint_path},
     LogDir: {log_dir},
     Image_Path: {log_dir}/images/combined_{name}.png,
     EPOCHS: {EPOCHS}
     MAX_SAMPLES: {MAX_SAMPLES},
     MAX_LENGTH: {MAX_LENGTH},
     NUM_LAYERS: {NUM_LAYERS},
     D_MODEL: {D_MODEL},
     NUM_HEADS: {NUM_HEADS},
     UNITS: {UNITS},
     DROPOUT: {DROPOUT},
     BATCH_SIZE: {BATCH_SIZE},
     BUFFER_SIZE: {BUFFER_SIZE},
     VOCAB_SIZE: {VOCAB_SIZE},
{linebreak}"""
    with open("Parameters.txt", "a") as f:
        f.write(log)
    with open(f"{log_dir}/values/hparams.txt", "w", encoding="utf8") as f:
        data = f"""{str(MAX_SAMPLES)}
{name}
{str(MAX_LENGTH)}
{str(BATCH_SIZE)}
{str(BUFFER_SIZE)}
{str(NUM_LAYERS)}
{str(D_MODEL)}
{str(NUM_HEADS)}
{str(UNITS)}
{str(DROPOUT)}
{str(VOCAB_SIZE)}
{str(TARGET_VOCAB_SIZE)}
{str(MAX_SAMPLES)}
    """
        f.write(data)
        f.close()
    print("Done writing metadata")
    print("Writing Image Structure of the model")
    try:
        plot_model(model, f"{log_dir}/images/{name}_Image.png", expand_nested=True, show_shapes=True)
    except Exception as e:
        with open(f"{log_dir}/images/{name}_Image_Error.txt", "w") as f:
            f.write(f"Image error: {e}")
            print(f"Image error: {e}")
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch="500, 600")
    predict_callback = PredictCallback(tokenizer=tokenizer, start_token=START_TOKEN, end_token=END_TOKEN, max_length=MAX_LENGTH,
                                       log_dir=log_dir)
    print("Done.")
    print("Starting train....")

    model.compile(optimizer=optimizer, loss=gbf.loss_function, metrics=['accuracy'])
    with tf.profiler.experimental.Trace("Train"):
        model.fit(dataset_train, validation_data=dataset_val, epochs=EPOCHS,
                  callbacks=[cp_callback, predict_callback, tensorboard_callback], use_multiprocessing=True)
    print(log)
    print(linebreak)
    model.summary()
