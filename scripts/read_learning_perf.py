"""
Script for displaying the loss/dice values from a training experiments
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Display learning curves from experiments file (HDF5)."
    )
    parser.add_argument(
        "input_file", help="The experiments file (in hdf5 format)", type=str
    )
    parser.add_argument(
        "-p",
        "--position",
        help="Position of the text in the loss values figure",
        type=float,
        default=0.04,
    )
    args = parser.parse_args()

    font = {
        "family": "Helvetica",
        "color": "darkblue",
        "weight": "bold",
        "size": 30,
    }
    sns.set_theme(style="whitegrid")

    path = Path(__file__).parent.parent
    experiment = h5py.File(path / args.input_file, "r")
    exp_keys = list(experiment.keys())
    for exp in exp_keys:        
        train_loss = experiment[exp]["train_loss"]
        train_dice = experiment[exp]["train_dice"]
        val_loss = experiment[exp]["val_loss"]
        val_dice = experiment[exp]["val_dice"]
        test_loss = experiment[exp]["test_loss"][()]
        test_dice = experiment[exp]["test_dice"]

        loss = np.hstack((np.vstack(train_loss), np.vstack(val_loss)))
        dice = np.hstack((train_dice, val_dice))

        columns_loss = ["train_loss", "val_loss"]
        columns_dice = [
            "train_dice 0.2",
            "train_dice 0.4",
            "train_dice 0.6",
            "train_dice 0.8",
            "val_dice 0.2",
            "val_dice 0.4",
            "val_dice 0.6",
            "val_dice 0.8",
        ]
        data_loss = pd.DataFrame(loss, columns=columns_loss)
        data_dice = pd.DataFrame(dice, columns=columns_dice)
        fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10, 5))
        fig.suptitle("Performance in " + exp)
        sns.lineplot(ax=axes[0], data=data_loss, palette="rocket", linewidth=2.5)
        axes[0].text(60, 0.05, "test_loss = %.2f" % test_loss, fontdict=font)
        axes[0].set(xlim=(0, train_loss.shape[0]), xlabel="epochs")

        sns.lineplot(ax=axes[1], data=data_dice, palette="tab10", linewidth=2.5)
        plt.text(60, 0.4, "test_dice 0.2 = %.2f" % (test_dice[0]), fontdict=font)
        plt.text(60, 0.36, "test_dice 0.4 = %.2f" % (test_dice[1]), fontdict=font)
        plt.text(60, 0.32, "test_dice 0.6 = %.2f" % (test_dice[2]), fontdict=font)
        plt.text(60, 0.28, "test_dice 0.8 = %.2f" % (test_dice[3]), fontdict=font)
        axes[1].set(xlim=(0, train_loss.shape[0]), ylim=(0, 1), xlabel="epochs")

        plt.show()
