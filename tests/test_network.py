import numpy as np
import tensorflow as tf
from brainseg_fl.network import make_unet, segmentation_loss


def test_network_trains_and_reloads_weights(tmp_path):
    tf.random.set_seed(3)
    model = make_unet(32, width=2, depth=2, seed=3)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=segmentation_loss)
    rng = np.random.default_rng(3)
    x = rng.normal(size=(2, 32, 32, 1)).astype(np.float32)
    y = (x > 0.5).astype(np.float32)
    before = model.get_weights()
    result = model.train_on_batch(x, y, return_dict=True)
    assert np.isfinite(result["loss"])
    assert any(np.any(a != b) for a, b in zip(before, model.get_weights()))
    path = tmp_path / "model.weights.h5"; model.save_weights(str(path))
    other = make_unet(32, width=2, depth=2, seed=99); other.load_weights(str(path))
    np.testing.assert_allclose(model(x, training=False).numpy(), other(x, training=False).numpy(), rtol=1e-5, atol=1e-6)
