# plot_cifar_10.py
import os
import numpy as np
import matplotlib.pyplot as plt
from dataset import load_data

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def plot_examples(out_path="latex/images/cifar10_examples.png",
                  n_per_class=10, seed=0):
    train_images, train_labels, _, _ = load_data()    # (N,32,32,3) uint8, (N,)
    rng = np.random.default_rng(seed)

    fig, axes = plt.subplots(
        n_per_class, 10, figsize=(15, 1.9 * n_per_class + 0.3), squeeze=False
    )
    for c in range(10):
        idx = rng.choice(np.where(train_labels == c)[0], n_per_class, replace=False)
        for r, i in enumerate(idx):
            ax = axes[r, c]
            ax.imshow(train_images[i], interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(CLASSES[c], fontsize=20)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Gespeichert: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    plot_examples(n_per_class=4)