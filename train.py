import os
import shutil
import typing
import json
import numpy as np
import argparse as ap

os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

from GavinBackend.GavinCore.models import TransformerIntegration, tf, tfds, PerformerIntegration, FNetIntegration, PreTrainedEmbeddingTransformerIntegration, \
    RotaryTransformerIntegration
from GavinBackend.GavinCore.datasets import DatasetAPICreator, DatasetDirectFromFileAPICreator
from GavinBackend.GavinCore.load_data import load_tokenized_data
from GavinBackend.GavinCore.callbacks import PredictCallback

_MODEL_TYPES = {
    'transformer': TransformerIntegration,
    'performer': PerformerIntegration,
    'fnet': FNetIntegration,
    'rotary_transformer': RotaryTransformerIntegration,
    'pretrained_embedding_transformer': PreTrainedEmbeddingTransformerIntegration
}


def _get_embedding_idx(embedding_path):
    embedding_idx = {}
    with open(embedding_path, 'r', encoding='utf-8') as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embedding_idx[word] = coefs
    return embedding_idx


def _get_embedding_matrix(embedding_idx, tokenizer: tfds.deprecated.text.SubwordTextEncoder):
    i_dff = int(embedding_idx.get(list(embedding_idx.keys())[0]).shape[0])
    embedding_matrix = np.zeros((len(tokenizer.subwords) + 1, i_dff))
    for i, word in enumerate(tokenizer.subwords):
        embedding_vector = embedding_idx.get(word)
        if embedding_vector is not None and embedding_vector.shape[0] == i_dff:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix, i_dff


def _valid_path(string, bool_=False):
    if os.path.isfile(string):
        if bool_:
            return True
        return string
    elif bool_:
        return False
    else:
        raise FileNotFoundError(string)


def _dir_path(string):
    if "https://" in string:
        return string
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def _valid_args(args):
    if args.model == 'pretrained_embedding_transformer' and args.embedding_file is None:
        raise ValueError("Embedding file must be specified for pretrained_embedding_transformer")
    if args.model == 'performer' and args.features is None:
        raise ValueError("Number of features must be specified for performer")
    if not _valid_path(args.tokenizer_file + ".subwords") and not _valid_path(args.tokenizer_file + ".json"):
        raise FileNotFoundError(args.tokenizer_file)
    if "https" not in args.dataset_path:
        if not all(_valid_path(os.path.join(args.dataset_path, args.dataset_name + ext), bool_=True) for ext in ['.from', '.to']) and \
                not all(_valid_path(os.path.join(args.dataset_path, args.dataset_name + ext), bool_=True) for ext in ['-from.BIN', '-to.BIN']):
            raise FileNotFoundError(args.dataset_name)
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    return True


def _get_train_data(max_samples, dataset_path, file_name, model, buffer_size, batch_size, python_legacy=False, cpp_legacy=False,
                    use_memory_loaders=True):
    if python_legacy or use_memory_loaders:
        questions, answers = load_tokenized_data(max_samples=max_samples,
                                                 data_path=dataset_path,
                                                 filename=file_name,
                                                 s_token=model.start_token,
                                                 e_token=model.end_token, max_len=model.max_len,
                                                 python_legacy=python_legacy,
                                                 cpp_legacy=cpp_legacy)

        questions = tf.keras.preprocessing.sequence.pad_sequences(questions, maxlen=model.max_len, padding='post')
        answers = tf.keras.preprocessing.sequence.pad_sequences(answers, maxlen=model.max_len, padding='post')
        d_t, d_v = DatasetAPICreator.create_data_objects(questions, answers, buffer_size=buffer_size,
                                                         batch_size=batch_size,
                                                         vocab_size=model.vocab_size)
    else:
        path_to = os.path.join(dataset_path, "{}-{}.BIN")
        # noinspection StrFormat
        d_t, d_v = DatasetDirectFromFileAPICreator.create_data_objects(questions_file=path_to.format(file_name, "from"),
                                                                       answers_file=path_to.format(file_name, "to"),
                                                                       buffer_size=buffer_size,
                                                                       batch_size=batch_size,
                                                                       vocab_size=model.vocab_size,
                                                                       max_length=model.max_len,
                                                                       number_of_samples=max_samples,
                                                                       start_token=model.start_token[0],
                                                                       end_token=model.end_token[0],
                                                                       padding_value=0)

    return d_t, d_v


def _validate_kwargs(kwargs: typing.Dict, model_type: str):
    must_have_keys = ['num_layers', 'units', 'd_model', 'num_heads', 'base_log_dir', 'dropout',
                      'max_len', 'tokenizer', 'name', 'save_freq', 'batch_size', 'mixed', 'metadata']
    _unique_keys = {'performer': ['num_features'],
                    'pretrained_embedding_transformer': ['embedding_matrix']}
    for key in must_have_keys:
        if key not in kwargs:
            raise ValueError(f"Missing key {key} in kwargs")
    if model_type in _unique_keys.keys():
        for key in _unique_keys[model_type]:
            if key not in kwargs:
                raise ValueError(f"Missing key {key} in kwargs for model type {model_type}")


def _load_metadata(metadata_path) -> typing.Dict:
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata


def _load_previous_if_exists(model_type: str, log_dir: str,
                             model_name: str) -> typing.Optional[typing.Tuple[
    typing.Union[TransformerIntegration, PerformerIntegration, FNetIntegration, PreTrainedEmbeddingTransformerIntegration, RotaryTransformerIntegration],
    typing.Dict]]:
    if _valid_path(os.path.join(log_dir, model_name), bool_=True):
        metadata_path = os.path.join(log_dir, model_name, "config/", "metadata.json")
        if not _valid_path(metadata_path, bool_=True):
            print("Metadata file not found, checking for config.json instead")
            metadata_path = os.path.join(os.path.dirname(metadata_path), "config.json")
            if not _valid_path(metadata_path, bool_=True):
                print("config.json not found, removing model, likely invalid.")
                shutil.rmtree(os.path.join(log_dir, model_name))

        print("Loading previous model")
        model = _MODEL_TYPES[model_type].load(log_dir, model_name)
        metadata = _load_metadata(metadata_path)
    else:
        model = None
        metadata = None
    return model, metadata


def _freq_type(arg: typing.Union[str, int]) -> typing.Union[str, int]:
    if type(arg) in [str, int]:
        return arg if not arg.isnumeric() else int(arg)


def main():
    parser = ap.ArgumentParser()
    parser.add_argument('--python-legacy', action=ap.BooleanOptionalAction, default=False, help="Use the legacy python implementation of the data loading")
    parser.add_argument('--cpp-legacy', action=ap.BooleanOptionalAction, default=False, help="Use the legacy C++ implementation of the data loading")
    parser.add_argument('-dp', '--dataset-path', type=_dir_path, required=True, help='Path to the dataset')
    parser.add_argument('-dn', '--dataset-name', type=str, required=True, help="Dataset name")  # dataset_file_name
    parser.add_argument('--model', type=str, default='transformer', choices=_MODEL_TYPES.keys(), help='Model type')
    parser.add_argument('--mixed-precision', action=ap.BooleanOptionalAction, default=False, help='Use mixed precision training')
    parser.add_argument('--logdir', type=str, default='./logs/', help='Directory to save logs and checkpoints')
    parser.add_argument('--model-name', type=str, required=True, help='Name of the model to be saved')
    parser.add_argument('--embedding-file', type=_valid_path, help='Path to embedding file (only for pretrained_embedding_transformer)')
    parser.add_argument('--max-samples', type=int, default=None, help='Maximum number of samples to load from dataset', required=True)
    parser.add_argument('--batch-size', type=int, help='Maximum batch size to use for training', required=True)
    parser.add_argument('--buffer-size', type=int, help='Maximum buffer size to use for training', default=20_000)
    parser.add_argument('--tokenizer-file', type=str, help='Tokenizer to use', required=True)
    parser.add_argument('--epochs', type=int, help="Number of epochs to train for", required=True)
    parser.add_argument('--max-seq-length', type=int, help='Maximum sequence length to use for training', required=True)
    parser.add_argument('--layers', type=int, help='Number of layers to use for training', required=True)
    parser.add_argument('--d-model', type=int, help='Model dimension to use for training', required=True)
    parser.add_argument('--heads', type=int, help='Number of heads to use for training', required=True)
    parser.add_argument('--dff', type=int, help='Feed forward dimension to use for training', required=True)
    parser.add_argument('--dropout', type=float, help='Dropout to use for training', required=True)
    parser.add_argument('--save-every', type=_freq_type, help='Save every n epochs/steps ("epoch" for epochs, or number for every n steps)', default='epoch')
    parser.add_argument('--features', type=int, help='Number of features to use for training (only for performer)')
    parser.add_argument('--streaming', action=ap.BooleanOptionalAction, default=False, help='Use streaming the dataset for training (slow, will increase step time)')

    args = parser.parse_args()
    _valid_args(args)
    tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(args.tokenizer_file)

    model_kwargs = {
        'num_layers': args.layers,
        'units': args.dff,
        'd_model': args.d_model,
        'num_heads': args.heads,
        'base_log_dir': args.logdir,
        'dropout': args.dropout,
        'max_len': args.max_seq_length,
        'tokenizer': tokenizer,
        'name': args.model_name,
        'save_freq': args.save_every,
        'batch_size': args.batch_size,
        'mixed': args.mixed_precision,
        'metadata': {'max_samples': args.max_samples, 'batch_size': args.batch_size, 'buffer_size': args.buffer_size}
    }

    if args.model == 'pretrained_embedding_transformer':
        matrix, d_model = _get_embedding_matrix(_get_embedding_idx(args.embedding_file), tokenizer)
        print(f"You selected {args.d_model} however a value of {d_model} was used for D_MODEL because the embedding file was {d_model} in size.")
        model_kwargs['embedding_matrix'] = matrix
        model_kwargs['d_model'] = d_model
    elif args.model == 'performer':
        model_kwargs['num_features'] = args.features
    train_model(args.model, model_kwargs, max_samples=args.max_samples, dataset_path=args.dataset_path, dataset_name=args.dataset_name,
                buffer_size=args.buffer_size, epochs=args.epochs, streaming=args.streaming, python_legacy=args.python_legacy, cpp_legacy=args.cpp_legacy,
                tokenizer=tokenizer, batch_size=args.batch_size)


def train_model(model_type: typing.AnyStr, model_kwargs: typing.Dict, max_samples: int,
                batch_size: int, buffer_size: int, dataset_path: typing.AnyStr, dataset_name: typing.AnyStr,
                python_legacy: bool, cpp_legacy: bool, streaming: bool, tokenizer: tfds.deprecated.text.SubwordTextEncoder,
                epochs: int):
    """
    Train a model
    :param model_type:
        The type of model to train
    :param model_kwargs:
        The kwargs to pass to the model
    :param max_samples:
        The maximum number of samples to load from the dataset
    :param batch_size:
        The batch size to use for training
    :param buffer_size:
        The buffer size to use for training
    :param dataset_path:
        The path to the dataset
    :param dataset_name:
        The name of the dataset
    :param python_legacy:
        Use the legacy python implementation of the data loading
    :param cpp_legacy:
        Use the legacy C++ implementation of the data loading
    :param streaming:
        Use streaming the dataset for training (slow, will increase step time)
    :param tokenizer:
        The tokenizer to use
    :param epochs:
        The number of epochs to train for
    :return:
    """
    _validate_kwargs(model_kwargs, model_type)

    if model_kwargs['mixed']:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    model, metadata = _load_previous_if_exists(model_type, log_dir=model_kwargs['base_log_dir'], model_name=model_kwargs['name'])
    if model is None:
        model = _MODEL_TYPES[model_type](**model_kwargs)
    if metadata:
        max_samples = metadata['max_samples']
        batch_size = metadata['batch_size']
        buffer_size = metadata['buffer_size']

    dataset_train, dataset_val = _get_train_data(max_samples=max_samples, batch_size=batch_size, buffer_size=buffer_size,
                                                 dataset_path=dataset_path, file_name=dataset_name, model=model,
                                                 python_legacy=python_legacy, cpp_legacy=cpp_legacy, use_memory_loaders=not streaming)

    callbacks = model.get_default_callbacks()
    callbacks.pop(1)
    callbacks.insert(1, tf.keras.callbacks.TensorBoard(log_dir=model.log_dir, update_freq=model.save_freq,
                                                       embeddings_metadata=os.path.join(model.log_dir, "metadata.tsv"),
                                                       profile_batch=(100, 110), embeddings_freq=5))
    callbacks.pop(2)
    callbacks.insert(2, PredictCallback(tokenizer=tokenizer, start_token=model.start_token, end_token=model.end_token,
                                        max_length=model.max_len, log_dir=model.log_dir, update_freq=model.save_freq,
                                        wrapper_model=model))

    callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True))

    model.fit(dataset_train, validation_dataset=dataset_val, epochs=epochs, callbacks=callbacks)
    model.model.summary()


if __name__ == '__main__':
    main()
