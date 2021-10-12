import abc
import os
import typing
import json
import tensorflow_datasets as tfds

from .layers import PositionalEncoding, MultiHeadAttention, GPUEnabledEmbedding, MultiHeadPreformerAttention
from .utils import tf
from .preprocessing.text import preprocess_sentence
from .callbacks import PredictCallback
from .losses import SparseCategoricalCrossentropy
from tensorboard.plugins import projector


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model: int, warmup_steps: int = 4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step: int):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        config = super(CustomSchedule, self).get_config()
        config.update({
            'd_model': self.d_model,
            'warmup_steps': self.warmup_steps
        })
        return config


class TransformerAbstract(abc.ABC):
    def __init__(self, num_layers: int, units: int, d_model: int, num_heads: int, dropout: float, batch_size: int,
                 max_len: int, base_log_dir: typing.AnyStr, tokenizer: tfds.deprecated.text.SubwordTextEncoder = None,
                 name: typing.AnyStr = "transformer", mixed: bool = False, epochs: int = 0,
                 warmup_steps_learning_rate: int = 4000,
                 save_freq: typing.Union[int, typing.AnyStr] = 'epoch',
                 metadata=None, metrics: typing.List = None):
        if metrics is None:
            self.metrics = ['accuracy']
        else:
            self.metrics = metrics
        self.num_layers = num_layers
        self.units = units
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.start_token, self.end_token = [self.tokenizer.vocab_size + 1], [self.tokenizer.vocab_size + 2]
        self.vocab_size = self.tokenizer.vocab_size + 2
        self.default_dtype = tf.float32 if not mixed else tf.float16
        self.save_freq = save_freq
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps_learning_rate
        self.model = None

        self.name = name
        self.log_dir = os.path.join(base_log_dir, self.name)

        dirs_needed = ['images', 'tokenizer', 'config']
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        for dir_needed in dirs_needed:
            if not os.path.exists(os.path.join(self.log_dir, dir_needed)):
                os.mkdir(os.path.join(self.log_dir, dir_needed))

        self.config = {
            'NUM_LAYERS': self.num_layers,
            'UNITS': self.units,
            'D_MODEL': self.d_model,
            'NUM_HEADS': self.num_heads,
            'DROPOUT': self.dropout,
            'MAX_LENGTH': self.max_len,
            'TOKENIZER': self.tokenizer,
            'MODEL_NAME': self.name,
            'FLOAT16': True if self.default_dtype == tf.float16 else False,
            'EPOCHS': epochs,
            'SAVE_FREQ': save_freq,
            'BATCH_SIZE': batch_size
        }
        if metadata is None:
            metadata = {}
        self.metadata = metadata

        self.scce = tf.keras.losses.SparseCategoricalCrossentropy(
            reduction='none', from_logits=True)

        self.setup_model()

    @abc.abstractmethod
    def setup_model(self):
        raise NotImplementedError("Method not implemented.")

    @abc.abstractmethod
    def encoder_layer(self, name: str):
        raise NotImplementedError("Method not implemented.")

    @staticmethod
    @abc.abstractmethod
    def create_padding_mask(x):
        raise NotImplementedError("Method not implemented.")

    @abc.abstractmethod
    def create_look_ahead_mask(self, x) -> tf.Tensor:
        raise NotImplementedError("Method not implemented.")

    @abc.abstractmethod
    def encoder(self, name: str) -> tf.keras.Model:
        raise NotImplementedError("Method not implemented.")

    @abc.abstractmethod
    def decoder_layer(self, name: str) -> tf.keras.Model:
        raise NotImplementedError("Method not implemented.")

    @abc.abstractmethod
    def decoder(self, name: str) -> tf.keras.Model:
        raise NotImplementedError("Method not implemented.")

    def get_hparams(self) -> typing.Dict:
        return self.config

    def get_model(self) -> tf.keras.Model:
        return self.model

    def get_metadata(self) -> typing.Dict:
        return self.metadata

    def get_tokens(self) -> typing.Tuple[typing.List, typing.List]:
        """Return Start and End Tokens."""
        return self.start_token, self.end_token

    def get_optimizer(self) -> tf.keras.optimizers.Adam:
        learning_rate = CustomSchedule(self.d_model, warmup_steps=self.warmup_steps)
        return tf.keras.optimizers.Adam(learning_rate, beta_1=0.91, beta_2=0.98, epsilon=1e-9, clipnorm=5.0)

    def get_default_callbacks(self) -> typing.List:
        return [
            tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(self.log_dir, 'cp.ckpt'), save_weights_only=True,
                                               verbose=1, save_freq=self.save_freq),
            tf.keras.callbacks.TensorBoard(log_dir=self.log_dir),
            PredictCallback(tokenizer=self.tokenizer, start_token=self.start_token, end_token=self.end_token,
                            max_length=self.max_len,
                            log_dir=self.log_dir, wrapper_model=self)]

    def loss_function(self, y_true, y_pred) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        loss = self.scce(y_true, y_pred)
        # loss = SparseCategoricalCrossentropy(y_true, y_pred)

        mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
        loss = tf.multiply(loss, mask)

        return tf.reduce_mean(loss)

    def evaluate(self, sentence: typing.AnyStr) -> tf.Tensor:
        sentence = preprocess_sentence(sentence)

        sentence = tf.expand_dims(self.start_token + self.tokenizer.encode(sentence) + self.end_token, axis=0)

        output = tf.expand_dims(self.start_token, 0)

        for i in range(self.max_len):
            predictions = self.model(inputs=[sentence, output], training=False)

            # select the last word from the seq length dimension
            predictions = predictions[:, -1:, :]
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            if tf.equal(predicted_id, self.end_token[0]):
                break

            # concatenated the predicted_id to the output which is given the decoder
            # as its input
            output = tf.concat([output, predicted_id], axis=-1)
        return tf.squeeze(output, axis=0)

    def accuracy(self, y_true, y_pred) -> tf.Tensor:
        # ensure labels have shape (batch_size, MAX_LENGTH - 1)
        y_true = tf.reshape(y_true, shape=(-1, self.max_len - 1))
        return tf.metrics.SparseCategoricalAccuracy()(y_true, y_pred)

    def predict(self, sentence: str) -> typing.AnyStr:
        prediction = self.evaluate(sentence)

        predicated_sentence = self.tokenizer.decode([i for i in prediction if i < self.tokenizer.vocab_size])

        return predicated_sentence

    def compile(self) -> None:
        """Compile the model attribute to allow for training."""
        self.model.compile(optimizer=self.get_optimizer(), loss=self.loss_function, metrics=self.metrics)

    def save_hparams(self):
        # Saving config
        hparams = self.get_hparams()
        metadata = self.get_metadata()
        # Set the tokenizer to the save path not the object
        hparams['TOKENIZER'] = os.path.join(self.log_dir, os.path.join('tokenizer', self.name + '_tokenizer'))
        # Save the tokenizer
        self.tokenizer.save_to_file(os.path.join(self.log_dir, os.path.join('tokenizer', self.name + '_tokenizer')))
        file = open(os.path.join(self.log_dir, os.path.join('config', 'config.json')), 'w')
        json.dump(hparams, file)
        file.close()
        file = open(os.path.join(self.log_dir, os.path.join('config', 'metadata.json')), 'w')
        json.dump(metadata, file)
        file.close()

        # Writing Projector metadata for viewing in tensorboard
        with open(os.path.join(self.log_dir, 'metadata.tsv'), "w", encoding="utf-8") as f:
            for subwords in self.tokenizer.subwords:
                f.write(f"{subwords}\n")
            for unknown in range(1, self.tokenizer.vocab_size - len(self.tokenizer.subwords)):
                f.write(f"unknown #{unknown}\n")

        projector_config = projector.ProjectorConfig()
        embedding = projector_config.embeddings.add()

        embedding.metadata_path = 'metadata.tsv'
        projector.visualize_embeddings(self.log_dir, projector_config)

    @classmethod
    def load_model(cls, models_path, model_name):
        file = open(os.path.join(os.path.join(models_path, model_name), os.path.join('config', 'config.json')))
        # Prep the hparams for loading.
        hparams = json.load(file)
        file.close()
        tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(
            os.path.join(models_path, os.path.join(model_name, f'tokenizer/{model_name}_tokenizer')))
        hparams['TOKENIZER'] = tokenizer
        hparams = {k.lower(): v for k, v in hparams.items()}
        hparams['max_len'] = hparams['max_length']
        hparams['name'] = hparams['model_name']
        hparams['mixed'] = hparams['float16']
        hparams['base_log_dir'] = models_path
        del hparams['max_length'], hparams['model_name'], hparams['float16']

        base = cls(**hparams)
        base.get_model().load_weights(os.path.join(base.log_dir, 'cp.ckpt')).expect_partial()
        return base

    def fit(self, training_dataset: tf.data.Dataset, epochs: int,
            callbacks: typing.List = None, validation_dataset: tf.data.Dataset = None) -> tf.keras.callbacks.History:
        """Call .fit() on the model attribute.
        Runs the train sequence for self.model"""
        self.compile()
        try:
            tf.keras.utils.plot_model(self.model,
                                      to_file=os.path.join(os.path.join(self.log_dir, 'images'), 'image.png'),
                                      expand_nested=True,
                                      show_shapes=True, show_layer_names=True, show_dtype=True)
        except Exception as e:
            with open(os.path.join(os.path.join(self.log_dir, 'images'), 'error.txt'), 'w') as f:
                f.write(f"Image error: {e}")
                print(f"Image error: {e}")
                f.close()
        initial_epoch = self.config['EPOCHS']
        self.config['EPOCHS'] = self.config['EPOCHS'] + epochs
        self.save_hparams()
        with tf.profiler.experimental.Trace("Train"):
            history = self.model.fit(training_dataset, validation_data=validation_dataset, epochs=self.config['EPOCHS'],
                                     callbacks=callbacks if callbacks is not None else self.get_default_callbacks(),
                                     use_multiprocessing=True, initial_epoch=initial_epoch)
            return history


class TransformerIntegration(TransformerAbstract):
    """TransformerIntegration Model

    Based off paper: https://arxiv.org/pdf/1706.03762.pdf
    ...

    Attributes:
        :arg vocab_size: int
            The vocabulary size.
        :arg num_layers: int
            The Number of Encoder/Decoder Layers that the model has.
        :arg units: int
            ("dff" in paper), number of units the PointWiseFeedForward networks have/
        :arg d_model: int
            Representation Dimension
        :arg num_heads: int
            Number of Heads the Attention Mechanism has.
        :arg dropout: float
            Dropout value for dropout layers.
        :arg max_len: int
            Max Length of Sequence.
        :arg name: str
            Name Of Model.
    """

    def setup_model(self):
        inputs = tf.keras.Input(shape=(None,), name="inputs")
        dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

        enc_padding_mask = tf.keras.layers.Lambda(
            self.create_padding_mask, output_shape=(1, 1, None),
            name='enc_padding_mask')(inputs)
        # mask the future tokens for decoder inputs at the 1st attention block
        look_ahead_mask = tf.keras.layers.Lambda(
            self.create_look_ahead_mask,
            output_shape=(1, None, None),
            name='look_ahead_mask')(dec_inputs)
        # mask the encoder outputs for the 2nd attention block
        dec_padding_mask = tf.keras.layers.Lambda(
            self.create_padding_mask, output_shape=(1, 1, None),
            name='dec_padding_mask')(inputs)

        enc_outputs = self.encoder()(inputs=[inputs, enc_padding_mask])

        dec_outputs = self.decoder()(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

        outputs = tf.keras.layers.Dense(units=self.vocab_size, name="outputs", activation='softmax')(dec_outputs)

        self.model = tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=self.name)

    def encoder_layer(self, name: str = "encoder_layer") -> tf.keras.Model:
        """Encoder Layer
        Arguments:
            :arg name: str
                The name for the layer, returned in model.summary()
        """
        inputs = tf.keras.Input(shape=(None, self.d_model), name="inputs")
        padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

        attention = MultiHeadAttention(
            self.d_model, self.num_heads, name="attention")({'query': inputs,
                                                             'key': inputs,
                                                             'value': inputs,
                                                             'mask': padding_mask})
        attention = tf.keras.layers.Dropout(rate=self.dropout)(attention)
        attention = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)(inputs + attention)

        outputs = tf.keras.layers.Dense(units=self.units, activation='relu')(attention)
        outputs = tf.keras.layers.Dense(units=self.d_model)(outputs)
        outputs = tf.keras.layers.Dropout(rate=self.dropout)(outputs)
        outputs = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)(attention + outputs)

        return tf.keras.Model(
            inputs=[inputs, padding_mask], outputs=outputs, name=name)

    @staticmethod
    def create_padding_mask(x) -> tf.keras.Model:
        """Create a padding mask

        Mask the outputs for attention layers"""
        mask = tf.cast(tf.math.equal(x, 0), tf.float32)
        # batch_size, 1, 1, sequence_length
        return mask[:, tf.newaxis, tf.newaxis, :]

    def create_look_ahead_mask(self, x) -> tf.Tensor:
        """Create a Look Ahead mask

        Allows to "look" ahead into the sentence and make predictions based on that."""
        seq_len = tf.shape(x)[1]
        look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        padding_mask = self.create_padding_mask(x)
        return tf.maximum(look_ahead_mask, padding_mask)

    def encoder(self, name: str = 'encoder') -> tf.keras.Model:
        """Encoder Sub Model

        Arguments:
            :arg name: str
                The name for the sub model
        """
        inputs = tf.keras.Input(shape=(None,), name="inputs")
        padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

        embeddings = GPUEnabledEmbedding(self.vocab_size, self.d_model)(inputs)
        embeddings *= tf.math.sqrt(tf.cast(self.d_model, self.default_dtype))
        embeddings = tf.cast(embeddings, tf.float32)
        embeddings = PositionalEncoding(self.vocab_size, self.d_model)(embeddings)

        outputs = tf.keras.layers.Dropout(rate=self.dropout)(embeddings)

        for i in range(self.num_layers):
            outputs = self.encoder_layer(
                name="encoder_layer_{}".format(i),
            )([outputs, padding_mask])

        return tf.keras.Model(
            inputs=[inputs, padding_mask], outputs=outputs, name=name)

    def decoder_layer(self, name: str = "decoder_layer") -> tf.keras.Model:
        """Decoder Layer
                Arguments:
                    :arg name: str
                        The name for the layer, returned in model.summary()
                """
        inputs = tf.keras.Input(shape=(None, self.d_model), name="inputs")
        enc_outputs = tf.keras.Input(shape=(None, self.d_model), name="encoder_outputs")
        look_ahead_mask = tf.keras.Input(
            shape=(1, None, None), name="look_ahead_mask")
        padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

        attention1 = MultiHeadAttention(
            self.d_model, self.num_heads, name="attention_1")(inputs={'query': inputs,
                                                                      'key': inputs,
                                                                      'value': inputs,
                                                                      'mask': look_ahead_mask})
        attention1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)(attention1 + inputs)

        attention2 = MultiHeadAttention(
            self.d_model, self.num_heads, name="attention_2")(inputs={'query': attention1,
                                                                      'key': enc_outputs,
                                                                      'value': enc_outputs,
                                                                      'mask': padding_mask})
        attention2 = tf.keras.layers.Dropout(rate=self.dropout)(attention2)
        attention2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)(attention2 + attention1)

        outputs = tf.keras.layers.Dense(units=self.units, activation='relu')(attention2)
        outputs = tf.keras.layers.Dense(units=self.d_model)(outputs)
        outputs = tf.keras.layers.Dropout(rate=self.dropout)(outputs)
        outputs = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)(outputs + attention2)

        return tf.keras.Model(
            inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
            outputs=outputs,
            name=name)

    def decoder(self, name: str = 'decoder') -> tf.keras.Model:
        """Decoder Sub Model

        Arguments:
            :arg name: str
                The name for the sub model"""
        inputs = tf.keras.Input(shape=(None,), name='inputs')
        enc_outputs = tf.keras.Input(shape=(None, self.d_model), name='encoder_outputs')
        look_ahead_mask = tf.keras.Input(
            shape=(1, None, None), name='look_ahead_mask')
        padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

        embeddings = GPUEnabledEmbedding(self.vocab_size, self.d_model)(inputs)
        embeddings *= tf.math.sqrt(tf.cast(self.d_model, self.default_dtype))
        embeddings = tf.cast(embeddings, tf.float32)
        embeddings = PositionalEncoding(self.vocab_size, self.d_model)(embeddings)

        outputs = tf.keras.layers.Dropout(rate=self.dropout)(embeddings)

        for i in range(self.num_layers):
            outputs = self.decoder_layer(name='decoder_layer_{}'.format(i),
                                         )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

        return tf.keras.Model(
            inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
            outputs=outputs,
            name=name)


class PerformerIntegration(TransformerIntegration):
    """Improvement upon the original Transformer,
    the preformer seeks to greatly decrease the time and memory
    complexity of the original transformer model in terms of
    sequence length."""

    def __init__(self, num_layers: int, units: int, d_model: int, num_heads: int, dropout: float, max_len: int,
                 num_features: int, base_log_dir: typing.AnyStr, batch_size: int,
                 tokenizer: tfds.deprecated.text.SubwordTextEncoder = None,
                 name: typing.AnyStr = "performer", mixed: bool = False, epochs: int = 0,
                 warmup_steps_learning_rate: int = 4000,
                 save_freq: typing.Union[int, typing.AnyStr] = 'epoch',
                 metadata=None, metrics: typing.List = None):
        if num_features > d_model:
            raise ValueError(f"Value for Num_Features {num_features} must be LESS THAN or EQUAL to d_model {d_model}")

        self.num_features = num_features
        super(PerformerIntegration, self).__init__(num_layers=num_layers, units=units, d_model=d_model,
                                                   num_heads=num_heads, dropout=dropout, batch_size=batch_size,
                                                   max_len=max_len, base_log_dir=base_log_dir, tokenizer=tokenizer,
                                                   name=name, mixed=mixed, epochs=epochs, save_freq=save_freq,
                                                   metadata=metadata, metrics=metrics,
                                                   warmup_steps_learning_rate=warmup_steps_learning_rate)
        self.config['NUM_FEATURES'] = self.num_features

    def encoder_layer(self, name: str = "encoder_layer") -> tf.keras.Model:
        """Encoder Layer
                Arguments:
                    :arg name: str
                        The name for the layer, returned in model.summary()
                """
        inputs = tf.keras.Input(shape=(None, self.d_model), name="inputs")
        padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")
        attention = MultiHeadPreformerAttention(
            self.d_model, self.num_heads, self.num_features, name="attention")({'query': inputs,
                                                                                'key': inputs,
                                                                                'value': inputs,
                                                                                'mask': padding_mask})
        attention = tf.keras.layers.Dropout(rate=self.dropout)(attention)
        attention = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)(inputs + attention)

        outputs = tf.keras.layers.Dense(units=self.units, activation='relu')(attention)
        outputs = tf.keras.layers.Dense(units=self.d_model)(outputs)
        outputs = tf.keras.layers.Dropout(rate=self.dropout)(outputs)
        outputs = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)(attention + outputs)

        return tf.keras.Model(
            inputs=[inputs, padding_mask], outputs=outputs, name=name)

    def decoder_layer(self, name: str = "decoder_layer") -> tf.keras.Model:
        """Decoder Layer
                        Arguments:
                            :arg name: str
                                The name for the layer, returned in model.summary()
                        """
        inputs = tf.keras.Input(shape=(None, self.d_model), name="inputs")
        enc_outputs = tf.keras.Input(shape=(None, self.d_model), name="encoder_outputs")
        look_ahead_mask = tf.keras.Input(
            shape=(1, None, None), name="look_ahead_mask")
        padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')
        attention1 = MultiHeadPreformerAttention(
            self.d_model, self.num_heads, self.num_features, name="attention_1")(inputs={'query': inputs,
                                                                                         'key': inputs,
                                                                                         'value': inputs,
                                                                                         'mask': look_ahead_mask})
        attention1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)(attention1 + inputs)

        attention2 = MultiHeadPreformerAttention(
            self.d_model, self.num_heads, self.num_features, name="attention_2")(inputs={'query': attention1,
                                                                                         'key': enc_outputs,
                                                                                         'value': enc_outputs,
                                                                                         'mask': padding_mask})
        attention2 = tf.keras.layers.Dropout(rate=self.dropout)(attention2)
        attention2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)(attention2 + attention1)
        outputs = tf.keras.layers.Dense(units=self.units, activation='relu')(attention2)
        outputs = tf.keras.layers.Dense(units=self.d_model)(outputs)
        outputs = tf.keras.layers.Dropout(rate=self.dropout)(outputs)
        outputs = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)(outputs + attention2)

        return tf.keras.Model(
            inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
            outputs=outputs,
            name=name)

    def evaluate(self, sentence: typing.AnyStr) -> tf.Tensor:
        sentence = preprocess_sentence(sentence)

        sentence = tf.expand_dims(self.start_token + self.tokenizer.encode(sentence) + self.end_token, axis=0)

        output = tf.expand_dims(self.start_token, 0)
        sentence = tf.keras.preprocessing.sequence.pad_sequences(sentence, maxlen=self.max_len, padding='post')

        for i in range(self.max_len - 1):
            predictions = self.model(inputs=[sentence,
                                             tf.keras.preprocessing.sequence.pad_sequences(output, maxlen=self.max_len,
                                                                                           padding='post')],
                                     training=False)

            # select the last word from the seq length dimension
            predictions = predictions[:, -1:, :]
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            if tf.equal(predicted_id, self.end_token[0]):
                break

            # concatenated the predicted_id to the output which is given the decoder
            # as its input
            output = tf.concat([output, predicted_id], axis=-1)
        return tf.squeeze(output, axis=0)
