from GavinBackend.utils.model_management import *
from GavinBackend.learning_rate import CustomSchedule
from GavinBackend.functions import loss_function


def test_model_train():
    try:
        model, dataset_t, dataset_v, hparams = model_load("models", "D:\\Datasets\\reddit_data\\files\\", "Gerald_V3")
        learning_rate = CustomSchedule(hparams['D_MODEL'])
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.91, beta_2=0.98, epsilon=1e-9)
        model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
        model.fit(dataset_t, validation_data=dataset_v, epochs=1, use_multiprocessing=True)
        model.summary()
    except tf.errors.ResourceExhaustedError as e:
        assert False, f"Resources Run Out: {e}"
    except Exception as e:
        assert False, f"Error: {e}"
    finally:
        del model, dataset_t, dataset_v, hparams
