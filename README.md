# Neural Network Acceleration with Efficient Data Structures (TensorFlow)

This project demonstrates building performant TensorFlow training pipelines using efficient data structures and input pipelines (tf.data), along with model architectures that balance accuracy and throughput.

Key focuses:
- tf.data with caching, parallel mapping, vectorized preprocessing, and prefetching.
- Mixed precision and distribution strategy (if supported by your hardware).
- Lightweight CNNs with separable convolutions for speed.
- Simple experiment runner to compare input pipeline options.

## Quickstart

Prerequisites:
- Python 3.9–3.11
- TensorFlow 2.12–2.17 (CPU or GPU build)

Install dependencies:
```bash
pip install -r requirements.txt
```

Train a model (MNIST):
```bash
python src/train.py --dataset mnist --epochs 5 --batch-size 128 --augment --cache --prefetch --model separable
```

View training logs with TensorBoard:
```bash
tensorboard --logdir logs
```

Run multiple experiments:
```bash
python run_experiments.py
```

## Project Structure
```
.
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── run_experiments.py
└── src
    ├── __init__.py
    ├── data.py
    ├── models.py
    ├── train.py
    └── utils.py
```

## Datasets

Supported out-of-the-box:
- mnist
- fashion_mnist
- cifar10

You can extend src/data.py to add custom datasets.

## Tips for Performance

- Enable mixed precision on GPUs/Apple Silicon:
  ```bash
  python src/train.py --mixed-precision
  ```
- Use larger batch sizes if you have GPU memory headroom.
- Keep the input pipeline on GPU saturated with `--cache` and `--prefetch`.
- Prefer vectorized ops inside `tf.data` over Python loops.
- Profile with TensorBoard (Performance tab) to find bottlenecks.

## License

MIT – see LICENSE.