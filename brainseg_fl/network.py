"""A functional U-Net implementation of the published encoder/decoder method.

Reference: Ronneberger, Fischer and Brox (2015); see REFERENCES.md.
"""
import tensorflow as tf


def make_unet(image_size, width=16, depth=4, seed=42):
    if image_size % (2 ** depth):
        raise ValueError("Image size must be divisible by 2**depth")
    serial = iter(range(seed, seed + 1000))

    def pair(tensor, channels, label):
        for number in range(2):
            tensor = tf.keras.layers.Conv2D(
                channels, 3, padding="same", activation="relu", data_format="channels_last",
                kernel_initializer=tf.keras.initializers.HeNormal(seed=next(serial)),
                name=f"{label}_conv_{number + 1}")(tensor)
        return tensor

    inputs = tf.keras.Input((image_size, image_size, 1), name="flair_slice")
    tensor, shortcuts = inputs, []
    for level in range(depth):
        tensor = pair(tensor, width * 2 ** level, f"encoder_{level}")
        shortcuts.append(tensor)
        tensor = tf.keras.layers.MaxPooling2D(2, data_format="channels_last", name=f"down_{level}")(tensor)
    tensor = pair(tensor, width * 2 ** depth, "bridge")
    tensor = tf.keras.layers.SpatialDropout2D(0.2, data_format="channels_last", seed=next(serial), name="bridge_dropout")(tensor)
    for level in reversed(range(depth)):
        tensor = tf.keras.layers.Conv2DTranspose(
            width * 2 ** level, 2, strides=2, padding="same", data_format="channels_last",
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=next(serial)),
            name=f"up_{level}")(tensor)
        tensor = tf.keras.layers.Concatenate(axis=-1, name=f"skip_{level}")([tensor, shortcuts[level]])
        tensor = pair(tensor, width * 2 ** level, f"decoder_{level}")
    probabilities = tf.keras.layers.Conv2D(
        1, 1, activation="sigmoid", data_format="channels_last",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=next(serial)), name="tumour_probability")(tensor)
    return tf.keras.Model(inputs, probabilities, name=f"brainseg_unet_w{width}_d{depth}")


def segmentation_loss(truth, probability):
    truth = tf.cast(truth, tf.float32)
    axes = (1, 2, 3)
    overlap = tf.reduce_sum(truth * probability, axis=axes)
    total = tf.reduce_sum(truth + probability, axis=axes)
    soft_dice = (2.0 * overlap + 1.0) / (total + 1.0)
    cross_entropy = tf.reduce_mean(tf.keras.losses.binary_crossentropy(truth, probability))
    return 0.5 * cross_entropy + 0.5 * (1.0 - tf.reduce_mean(soft_dice))
