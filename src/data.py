from typing import Tuple, Optional

import tensorflow as tf


AUTOTUNE = tf.data.AUTOTUNE

def _load_builtin_dataset(name: str):
    name = name.lower()
    if name == "mnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train[..., None]
        x_test = x_test[..., None]
        num_classes = 10
    elif name == "fashion_mnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        x_train = x_train[..., None]
        x_test = x_test[..., None]
        num_classes = 10
    elif name == "cifar10":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        y_train = y_train.squeeze().astype("int64")
        y_test = y_test.squeeze().astype("int64")
        num_classes = 10
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    input_shape = x_train.shape[1:]
    return (x_train, y_train), (x_test, y_test), input_shape, num_classes

def _augment_example(image: tf.Tensor, label: tf.Tensor, channels: int) -> Tuple[tf.Tensor, tf.Tensor]:
    # Simple, fast augmentations that are GPU-friendly
    if channels == 1:
        # Convert to 3 channels for consistent ops if needed
        image = tf.image.grayscale_to_rgb(image)
        channels = 3

    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.clip_by_value(image, 0.0, 1.0)
    # Return to original channels if we expanded
    if channels == 3 and image.shape[-1] == 3 and tf.shape(label)[0:0].shape.rank is not None:
        pass
    return image, label

def _preprocess(example: Tuple[tf.Tensor, tf.Tensor], num_classes: int, augment: bool, channels: int):
    image, label = example
    image = tf.cast(image, tf.float32) / 255.0
    if augment:
        image, label = _augment_example(image, label, channels)
    label = tf.cast(label, tf.int32)
    label = tf.one_hot(label, depth=num_classes)
    return image, label

def build_dataset(
    x,
    y,
    batch_size: int,
    num_classes: int,
    cache: bool = True,
    shuffle: bool = True,
    augment: bool = False,
    prefetch: bool = True,
    deterministic: Optional[bool] = None,
) -> tf.data.Dataset:
    channels = x.shape[-1]
    ds = tf.data.Dataset.from_tensor_slices((x, y))

    if shuffle:
        # Larger buffer_size improves randomness at the cost of memory
        buffer_size = min(len(x), 10000)
        ds = ds.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)

    # Map preprocessing with vectorized ops and parallel calls
    ds = ds.map(
        lambda img, lbl: _preprocess((img, lbl), num_classes, augment, channels),
        num_parallel_calls=AUTOTUNE,
        deterministic=deterministic,
    )

    if cache:
        ds = ds.cache()

    ds = ds.batch(batch_size, drop_remainder=False)

    if prefetch:
        ds = ds.prefetch(AUTOTUNE)

    return ds

def get_datasets(
    name: str = "mnist",
    batch_size: int = 128,
    cache: bool = True,
    prefetch: bool = True,
    augment: bool = False,
    val_split: float = 0.1,
    deterministic: Optional[bool] = None,
):
    (x_train, y_train), (x_test, y_test), input_shape, num_classes = _load_builtin_dataset(name)

    # Train/Val split
    n_train = int(len(x_train) * (1 - val_split))
    x_tr, y_tr = x_train[:n_train], y_train[:n_train]
    x_val, y_val = x_train[n_train:], y_train[n_train:]

    train_ds = build_dataset(
        x_tr, y_tr, batch_size, num_classes, cache=cache, shuffle=True, augment=augment, prefetch=prefetch, deterministic=deterministic
    )
    val_ds = build_dataset(
        x_val, y_val, batch_size, num_classes, cache=True, shuffle=False, augment=False, prefetch=prefetch, deterministic=deterministic
    )
    test_ds = build_dataset(
        x_test, y_test, batch_size, num_classes, cache=True, shuffle=False, augment=False, prefetch=prefetch, deterministic=deterministic
    )

    steps_per_epoch = max(len(x_tr) // batch_size, 1)
    val_steps = max(len(x_val) // batch_size, 1)

    return {
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "steps_per_epoch": steps_per_epoch,
        "val_steps": val_steps,
        "input_shape": input_shape,
        "num_classes": num_classes,
    }
