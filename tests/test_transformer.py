import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from GavinCore import Transformer


def test_model_create():
    try:
        base = Transformer(vocab_size=8096, num_layers=2, units=512, d_model=256,
                           num_heads=8, dropout=0.1, name="TestTransformer")
        model = base.return_model()
    except Exception as e:
        assert False, f"Error: {e}"
    finally:
        try:
            del model
        except ReferenceError:
            pass
