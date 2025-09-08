import argparse
import os
from datetime import datetime
from typing import Dict, Any, Optional

import tensorflow as tf

from .data import get_datasets
from .models import build_model
from .utils import (
    seed_everything,
    configure_gpus,
    maybe_enable_mixed_precision,
    get_strategy,
    throughput_timer,
)

def compile_model(model: tf.keras.Model, learning_rate: float):
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.CategoricalCrossentropy()
    metrics = [
        tf.keras.metrics.CategoricalAccuracy(name="acc"),
        tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3"),
    ]
    model.compile(optimizer=opt, loss=loss, metrics=metrics)

def make_callbacks(log_dir: str, save_checkpoints: bool, early_stopping: bool, patience: int):
    os.makedirs(log_dir, exist_ok=True)
    cbs = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, profile_batch=0),
    ]
    if save_checkpoints:
        ckpt_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        cbs.append(tf.keras.callbacks.ModelCheckpoint(os.path.join(ckpt_dir, "best.keras"), monitor="val_acc", save_best_only=True))
    if early_stopping:
        cbs.append(tf.keras.callbacks.EarlyStopping(monitor="val_acc", patience=patience, restore_best_weights=True))
    return cbs

def run_training(config: Dict[str, Any]):
    dataset = config.get("dataset", "mnist")
    batch_size = int(config.get("batch_size", 128))
    epochs = int(config.get("epochs", 5))
    learning_rate = float(config.get("learning_rate", 1e-3))
    cache = bool(config.get("cache", True))
    prefetch = bool(config.get("prefetch", True))
    augment = bool(config.get("augment", False))
    model_name = str(config.get("model", "separable"))
    mixed_precision = bool(config.get("mixed_precision", False))
    log_dir = str(config.get("log_dir", os.path.join("logs", "runs", datetime.now().strftime("%Y%m%d-%H%M%S"))))
    run_name = str(config.get("run_name", "run"))
    save_checkpoints = bool(config.get("save_checkpoints", True))
    early_stopping = bool(config.get("early_stopping", True))
    patience = int(config.get("patience", 3))
    deterministic = config.get("deterministic", None)

    seed_everything(42)
    configure_gpus()
    maybe_enable_mixed_precision(mixed_precision)

    log_dir_run = os.path.join(log_dir, run_name)
    os.makedirs(log_dir_run, exist_ok=True)

    with get_strategy().scope():
        data_info = get_datasets(
            name=dataset,
            batch_size=batch_size,
            cache=cache,
            prefetch=prefetch,
            augment=augment,
            deterministic=deterministic,
        )

        model = build_model(model_name, data_info["input_shape"], data_info["num_classes"]) 
        compile_model(model, learning_rate)

    total_train_samples = data_info["steps_per_epoch"] * batch_size

    callbacks = make_callbacks(log_dir_run, save_checkpoints=save_checkpoints, early_stopping=early_stopping, patience=patience)

    print(f"[INFO] Starting training: {dataset} | model={model_name} | bs={batch_size} | cache={cache} | prefetch={prefetch} | augment={augment}")
    with throughput_timer(total_samples=total_train_samples * epochs):
        history = model.fit(
            data_info["train_ds"],
            validation_data=data_info["val_ds"],
            epochs=epochs,
            steps_per_epoch=data_info["steps_per_epoch"],
            validation_steps=data_info["val_steps"],
            callbacks=callbacks,
            verbose=2,
        )

    print("[INFO] Evaluating on test set...")
    test_metrics = model.evaluate(data_info["test_ds"], verbose=0, return_dict=True)
    print("[RESULT] Test metrics:", test_metrics)

    # Save final model
    export_path = os.path.join(log_dir_run, "final_model.keras")
    try:
        model.save(export_path)
        print(f"[INFO] Saved final model to {export_path}")
    except Exception as e:
        print(f"[WARN] Could not save model: {e}")

    return {
        "history": history.history if "history" in dir(history) else {},
        "test_metrics": test_metrics,
        "log_dir": log_dir_run,
        "export_path": export_path,
    }

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CNNs with efficient tf.data pipelines.")
    p.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist", "cifar10"])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--no-augment", dest="augment", action="store_false")
    p.set_defaults(augment=False)
    p.add_argument("--cache", action="store_true")
    p.add_argument("--no-cache", dest="cache", action="store_false")
    p.set_defaults(cache=True)
    p.add_argument("--prefetch", action="store_true")
    p.add_argument("--no-prefetch", dest="prefetch", action="store_false")
    p.set_defaults(prefetch=True)
    p.add_argument("--model", type=str, default="separable", choices=["simple", "separable"])
    p.add_argument("--mixed-precision", action="store_true")
    p.add_argument("--log-dir", type=str, default=os.path.join("logs", "runs"))
    p.add_argument("--run-name", type=str, default="run")
    p.add_argument("--save-checkpoints", action="store_true")
    p.add_argument("--no-save-checkpoints", dest="save_checkpoints", action="store_false")
    p.set_defaults(save_checkpoints=True)
    p.add_argument("--early-stopping", action="store_true")
    p.add_argument("--no-early-stopping", dest="early_stopping", action="store_false")
    p.set_defaults(early_stopping=True)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--deterministic", type=str, default="auto", help="Set to true/false/auto to control tf.data determinism.")
    return p.parse_args()

def _parse_bool_auto(val: str) -> Optional[bool]:
    v = str(val).strip().lower()
    if v in ("true", "t", "1", "yes", "y"):
        return True
    if v in ("false", "f", "0", "no", "n"):
        return False
    return None  # auto

def main():
    args = parse_args()
    cfg = {
        "dataset": args.dataset,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "augment": args.augment,
        "cache": args.cache,
        "prefetch": args.prefetch,
        "model": args.model,
        "mixed_precision": args.mixed_precision,
        "log_dir": args.log_dir,
        "run_name": args.run_name,
        "save_checkpoints": args.save_checkpoints,
        "early_stopping": args.early_stopping,
        "patience": args.patience,
        "deterministic": _parse_bool_auto(args.deterministic),
    }
    run_training(cfg)

if __name__ == "__main__":
    main()
