from GavinCore import tf
from GavinCore.layers import PositionalEncoding, MultiHeadAttention


class Transformer:
    """Transformer Model

    Based off paper: https://arxiv.org/pdf/1706.03762.pdf
    ...

    Attributes:
        :arg vocab_size: int
            The vocabulary size the Tokenizer Uses.
        :arg num_layers: int
            The Number of Encoder/Decoder Layers that the model has.
        :arg units: int
            The Number of units the Encoder/Decoder Layers have.
        :arg d_model: int
            Output Units for the Embedding Layers
        :arg num_heads: int
            Number of Heads the MultiHead attention will be configured with
        :arg dropout: float
            The Dropout that the model will have. This number is between 0 and 1. Do not go higher.
        :arg name: str
            The Name the model will be configured with, defaults to "transformer"
        :arg **kwargs
            Key Word arguments to pass to tf.keras.Model super class
    """

    def __init__(self, vocab_size: int, num_layers: int, units: int, d_model: int, num_heads: int, dropout: float,
                 name: str = "transformer", mixed: bool = False, **kwargs):
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.units = units
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self._model_name = name
        self.default_dtype = tf.float32 if not mixed else tf.float16
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

        outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)

        self.model = tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)

    def return_model(self):
        return self.model

    def encoder_layer(self, name: str = "encoder_layer"):
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
    def create_padding_mask(x):
        """Create a padding mask

        Mask the outputs for attention layers"""
        mask = tf.cast(tf.math.equal(x, 0), tf.float32)
        # batch_size, 1, 1, sequence_length
        return mask[:, tf.newaxis, tf.newaxis, :]

    def create_look_ahead_mask(self, x):
        """Create a Look Ahead mask

        Allows to "look" ahead into the sentence and make predictions based on that."""
        seq_len = tf.shape(x)[1]
        look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        padding_mask = self.create_padding_mask(x)
        return tf.maximum(look_ahead_mask, padding_mask)

    def encoder(self, name: str = 'encoder'):
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

    def decoder_layer(self, name: str = "decoder_layer"):
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

    def decoder(self, name: str = 'decoder'):
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
