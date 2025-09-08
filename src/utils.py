import os
import random
import time
from contextlib import contextmanager
from typing import Optional

import numpy as np
import tensorflow as tf

def seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def configure_gpus(memory_growth: bool = True):
    gpus = tf.config.list_physical_devices("GPU")
    if gpus and memory_growth:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

def maybe_enable_mixed_precision(enabled: bool):
    if not enabled:
        return
    try:
        from tensorflow.keras import mixed_precision
        policy = mixed_precision.Policy("mixed_float16")
        mixed_precision.set_global_policy(policy)
        print("[INFO] Mixed precision enabled:", mixed_precision.global_policy())
    except Exception as e:
        print("[WARN] Could not enable mixed precision:", e)

def get_strategy(distribute: bool = True) -> tf.distribute.Strategy:
    # MirroredStrategy will use all visible GPUs
    if distribute and len(tf.config.list_physical_devices("GPU")) > 0:
        return tf.distribute.MirroredStrategy()
    # Default strategy (no distribution)
    return tf.distribute.get_strategy()

@contextmanager
def throughput_timer(total_samples: int):
    start = time.time()
    yield
    end = time.time()
    elapsed = max(end - start, 1e-9)
    print(f"[METRIC] Elapsed: {elapsed:.2f}s, Samples: {total_samples}, Throughput: {total_samples/elapsed:.2f} samples/s")
