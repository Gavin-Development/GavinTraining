import numpy as np
import random
from GavinBackend.preprocessing.text import preprocess_sentence
from GavinBackend.models import tf


class PredictCallback(tf.keras.callbacks.Callback):
    def __init__(self, tokenizer, start_token, end_token, max_length, log_dir):
        super(PredictCallback, self).__init__()
        self.tokenizer = tokenizer
        self.START_TOKEN = start_token
        self.END_TOKEN = end_token
        self.MAX_LENGTH = max_length
        self.prompts = ["Hey?", "Hi?", "Hello.", "How are you?", "How are you doing?", "My name is Josh.",
                        "Nice to meet you!", "What is your name?", "How old are you?", "Are you married?"]
        random.shuffle(self.prompts)
        self.log_dir = log_dir

        self.title_formatting = "=" * 10
        self.past_tests = []
        self.past_logs = []

    def _evaluate(self, sentence):
        sentence = preprocess_sentence(sentence)

        sentence = tf.expand_dims(self.START_TOKEN + self.tokenizer.encode(sentence) + self.END_TOKEN, axis=0)

        output = tf.expand_dims(self.START_TOKEN, 0)

        for i in range(self.MAX_LENGTH):
            predictions = self.model(inputs=[sentence, output], training=False)

            # select the last word from the seq length dimension
            predictions = predictions[:, -1:, :]
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            if tf.equal(predicted_id, self.END_TOKEN[0]):
                break

            # concatenated the predicted_id to the output which is given the decoder
            # as its input
            output = tf.concat([output, predicted_id], axis=-1)
        return tf.squeeze(output, axis=0)

    def _predict(self):
        predictions = []
        print("Predicting... (This could take a little bit.)")
        for i in range(random.randint(4, len(self.prompts))):  # TODO convert this to gradually increase the number of tests maxing at a value set at construction time.
            sentence = self.prompts[i]
            prediction = self._evaluate(sentence)

            predictions.append(
                (sentence, self.tokenizer.decode([i for i in prediction if i < self.tokenizer.vocab_size])))
        random.shuffle(self.prompts)

        return predictions

    def file_writer(self):
        try:
            with open(f"{self.log_dir}/logs/chat_logs.log", "w") as f:
                for (sentence, response, epoch) in self.past_tests:
                    f.write(f"{self.title_formatting}\nEpoch:{epoch}\nInput:{sentence}\nOutput:{response}\n")
                f.close()
            with open(f"{self.log_dir}/logs/model_logs.log", "w") as f:
                for (epoch, logs) in self.past_tests:
                    f.write(f"{self.title_formatting}\nEpoch: {epoch}\nLogs: {logs}")
                f.close()
        except Exception as e:
            print(f"File writer failed because: {e}")
        print("Wrote Files.")

    def on_epoch_end(self, epoch, logs=None):
        epoch += 1
        tests = self._predict()
        print(f"{self.title_formatting} Responses for Epoch: {epoch} {self.title_formatting}")
        for (sentence, response) in tests:
            print(f"Input: {sentence}\nOutput: {response}")
            self.past_tests.append((sentence, response, epoch))
            self.past_logs.append((epoch, logs))
        if logs is not None:
            print(f"{self.title_formatting} Log Information {self.title_formatting}")
            keys = list(logs.keys())
            print(f"Log Keys: {keys}")
        print(self.title_formatting + self.title_formatting + self.title_formatting)

    def on_train_end(self, logs=None):
        self.file_writer()
        # Any other end jobs go here


# Source: https://www.tensorflow.org/guide/keras/custom_callback#usage_of_selfmodel_attribute
class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

      Arguments:
          patience: Number of epochs to wait after min has been hit. After this
          number of no improvement, training stops.
      """

    def __init__(self, patience=0):
        super(EarlyStoppingAtMinLoss, self).__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

        # Init self vars
        self.wait = 0
        self.stopped_epoch = 0
        self.best = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)
                self.model.save_weights(self)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))
