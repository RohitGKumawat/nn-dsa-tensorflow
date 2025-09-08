import itertools
import os
from datetime import datetime

from src.train import run_training

def main():
    # Grid of experiments focusing on input pipeline efficiency
    configs = {
        "dataset": ["mnist", "fashion_mnist", "cifar10"],
        "batch_size": [64, 128, 256],
        "cache": [True, False],
        "prefetch": [True, False],
        "augment": [True, False],
        "model": ["separable", "simple"],
    }

    base_logdir = os.path.join("logs", f"experiments_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(base_logdir, exist_ok=True)

    keys = list(configs.keys())
    for values in itertools.product(*[configs[k] for k in keys]):
        cfg = dict(zip(keys, values))
        run_name = (
            f"{cfg['dataset']}_bs{cfg['batch_size']}"
            f"_cache{int(cfg['cache'])}_pref{int(cfg['prefetch'])}"
            f"_aug{int(cfg['augment'])}_{cfg['model']}"
        )
        cfg.update(
            {
                "epochs": 3,
                "learning_rate": 1e-3,
                "mixed_precision": False,
                "log_dir": os.path.join(base_logdir, run_name),
                "run_name": run_name,
                "save_checkpoints": False,
                "early_stopping": True,
                "patience": 2,
            }
        )
        print(f"=== Running: {run_name} ===")
        run_training(cfg)

if __name__ == "__main__":
    main()
