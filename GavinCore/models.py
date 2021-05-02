import os
import typing

from GavinCore import tf, tfds
from GavinCore.layers import PositionalEncoding, MultiHeadAttention
from GavinCore.preprocessing.text import preprocess_sentence
from GavinCore.callbacks import PredictCallback


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model: int, warmup_steps: int = 5000):
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


class TransformerIntegration:
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

    def __init__(self, vocab_size: int, num_layers: int, units: int, d_model: int, num_heads: int, dropout: float,
                 max_len: int, base_log_dir: typing.AnyStr, tokenizer: tfds.deprecated.text.SubwordTextEncoder = None,
                 name: typing.AnyStr = "transformer", mixed: bool = False):
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.units = units
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_len = max_len
        self.start_token, self.end_token = [self.vocab_size], [self.vocab_size + 2]
        self.vocab_size += 2
        self.tokenizer = tokenizer
        self.tokenizer.vocab_size = self.vocab_size
        self.name = name
        self.log_dir = os.path.join(base_log_dir, self.name)
        self.default_dtype = tf.float32 if not mixed else tf.float16
        self.model = None

        self.setup_model()

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

        outputs = tf.keras.layers.Dense(units=self.vocab_size, name="outputs")(dec_outputs)

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

        embeddings = tf.keras.layers.Embedding(self.vocab_size, self.d_model)(inputs)
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

        embeddings = tf.keras.layers.Embedding(self.vocab_size, self.d_model)(inputs)
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

    def get_hparams(self) -> typing.Dict:
        config = {
            'VOCAB_SIZE': self.vocab_size,
            'NUM_LAYERS': self.num_layers,
            'UNITS': self.units,
            'D_MODEL': self.d_model,
            'NUM_HEADS': self.num_heads,
            'DROPOUT': self.dropout,
            'MODEL_NAME': self.name,
            'FLOAT16': True if self.default_dtype == tf.float16 else False
        }
        return config

    def get_model(self) -> tf.keras.Model:
        return self.model

    def get_tokens(self) -> typing.Tuple[typing.List, typing.List]:
        """Return Start and End Tokens."""
        return self.start_token, self.end_token

    def get_optimizer(self) -> tf.keras.optimizers.Adam:
        learning_rate = CustomSchedule(self.d_model)
        return tf.keras.optimizers.Adam(learning_rate, beta_1=0.91, beta_2=0.98, epsilon=1e-9)

    def get_default_callbacks(self) -> typing.List:
        return [
            tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(self.log_dir, 'cp.ckpt'), save_weights_only=True,
                                               verbose=1),
            tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, profile_batch="500, 600"),
            PredictCallback(tokenizer=self.tokenizer, start_token=self.start_token, end_token=self.end_token,
                            max_length=self.max_len,
                            log_dir=self.log_dir)]

    def loss_function(self, y_true, y_pred) -> tf.Tensor:
        y_true = tf.reshape(y_true, shape=(-1, self.max_len - 1))

        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')(y_true, y_pred)

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

        predicated_sentence = self.tokenizer.decode([i for i in prediction if i < self.vocab_size])

        return predicated_sentence

    def compile(self) -> None:
        """Compile the model attribute to allow for training."""
        self.model.compile(optimizer=self.get_optimizer(), loss=self.loss_function, metrics=['accuracy'])

    def fit(self, training_dataset: tf.data.Dataset,
            epochs: int,
            initial_epoch: int = 0,
            callbacks: typing.List = None,
            validation_dataset: tf.data.Dataset = None) -> tf.keras.callbacks.History:
        """Call .fit() on the model attribute.
        Runs the train sequence for self.model"""
        with tf.profiler.experimental.Trace("Train"):
            history = self.model.fit(training_dataset, validation_dataset=validation_dataset, epochs=epochs,
                                     callbacks=callbacks if callbacks is not None else self.get_default_callbacks(),
                                     use_multiprocessing=True, initial_epoch=initial_epoch)
            return history
