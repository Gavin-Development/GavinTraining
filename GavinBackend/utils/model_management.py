import json
import os

from GavinBackend import tfds, tf
from GavinBackend.preprocessing.text import load_tokenized_data
from GavinBackend.models import Transformer
from typing import Tuple, Union, OrderedDict


def set_policies(other_policy: bool) -> None:
    """Set the GPU policies
    Note: Mixed_Precision is Not yet supported."""
    mixed_precision = tf.keras.mixed_precision.experimental
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if other_policy:
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
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(f"{len(gpus)} Physical GPUS, {len(logical_gpus)} Logical GPUS.")
            except RuntimeError as e:
                print(e)


def model_save(hparams: Union[dict, OrderedDict], log_dir: str) -> None:
    try:
        os.mkdir(f"{log_dir}")
        os.mkdir(f"{log_dir}/tokenizer")
        os.mkdir(f"{log_dir}/values/")
        os.mkdir(f"{log_dir}/images/")
        os.mkdir(f"{log_dir}/logs/")
    except FileExistsError:
        pass
    output_dict = json.dumps(hparams)
    json.dump(fp=open(f"{log_dir}/values/config.json", "rb"), obj=output_dict)


def model_load(log_dir: str, dataset_path: str, model_name: str) -> Tuple[tf.keras.models.Model, tf.data.Dataset, tf.data.Dataset, dict]:
    hparams = json.load(fp=open(os.path.join(os.path.join(log_dir, model_name), os.path.join("values/", "config.json"))))
    print(f"Import Hyper Parameters: {hparams}")
    # (input("Are these correct? y/n: ").lower() in ["no", "n"] or print("Continuing")) and quit()
    return create_model(log_dir, dataset_path, hparams, load=True)


def create_model(log_dir: str, dataset_path: str, hparams: Union[dict, OrderedDict], load: bool = False, tokenizer_path: str = None) -> Tuple[tf.keras.models.Model, tf.data.Dataset, tf.data.Dataset, dict]:
    """Create a model, "transformer", and return it."""
    if not load and tokenizer_path is None:
        raise Exception("Either load=True or pass path to tokenizer_path.")
    set_policies(hparams['OTHER_POLICY'])

    mirrored_strategy = tf.distribute.MirroredStrategy()

    if load:
        loaded_tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(
            f"{log_dir}/{hparams['MODEL_NAME']}/tokenizer/vocabTokenizer")
    else:
        loaded_tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(tokenizer_path)

    start_token, end_token = [loaded_tokenizer.vocab_size], [
        loaded_tokenizer.vocab_size + 1]  # Set the START and END tokens

    questions, answers = load_tokenized_data(hparams["MAX_SAMPLES"], dataset_path, hparams['TOKENIZER_NAME'],
                                             hparams['MAX_LENGTH'], start_token, end_token)
    sizes = (len(questions), len(answers))
    questions_train = questions[0: int(sizes[0] * .80)]
    questions_val = questions[int(sizes[0] * 0.80):]
    answers_train = answers[0: int(sizes[0] * .80)]
    answers_val = answers[int(sizes[0] * 0.80):]
    dataset_t = tf.data.Dataset.from_tensor_slices((
        {
            'inputs': questions_train,  # Source
            'dec_inputs': answers_train[:, :-1]  # Targets
        },
        {
            'outputs': answers_train[:, 1:]  # Outputs
        }))
    dataset_v = tf.data.Dataset.from_tensor_slices((
        {
            'inputs': questions_val,  # Source
            'dec_inputs': answers_val[:, :-1]  # Targets
        },
        {
            'outputs': answers_val[:, 1:]  # Outputs
        }))

    dataset_t = dataset_t.cache()
    dataset_v = dataset_v.cache()
    dataset_t = dataset_t.shuffle(hparams["BUFFER_SIZE"])
    dataset_v = dataset_v.shuffle(hparams["BUFFER_SIZE"])
    dataset_t = dataset_t.batch(hparams["BUFFER_SIZE"])
    dataset_v = dataset_v.batch(hparams["BUFFER_SIZE"])
    dataset_t = dataset_t.prefetch(tf.data.experimental.AUTOTUNE)
    dataset_v = dataset_v.prefetch(tf.data.experimental.AUTOTUNE)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    dataset_t = dataset_t.with_options(options)
    dataset_v = dataset_v.with_options(options)

    with mirrored_strategy.scope():
        base = Transformer(vocab_size=hparams["VOCAB_SIZE"],
                           num_layers=hparams["NUM_LAYERS"],
                           units=hparams["UNITS"],
                           d_model=hparams["D_MODEL"],
                           num_heads=hparams["NUM_HEADS"],
                           dropout=hparams["DROPOUT"],
                           name=hparams["MODEL_NAME"],
                           mixed=hparams["OTHER_POLICY"])
        model = base.return_model()
        if os.path.exists(os.path.join(os.path.join(log_dir, hparams['MODEL_NAME']), "checkpoint")) and load:
            model.load_weights(os.path.join(os.path.join(log_dir, hparams['MODEL_NAME']), "cp.ckpt")).expect_partial()
    return model, dataset_t, dataset_v, hparams if load else None
