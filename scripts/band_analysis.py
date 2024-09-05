"""
Compare two segmentations results for filaments
"""
import argparse
import astropy.io.fits as fits
import numpy as np
from deep_filaments.metrics.metrics import dice_index
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Measure the quality of a segmentation."
    )
    parser.add_argument("segmentation", help="The segmentation result", type=str)
    parser.add_argument("spines", help="The groundtruth", type=str)
    parser.add_argument("background", help="The background", type=str)
    args = parser.parse_args()
    
    if args.segmentation.endswith(".fits"):
        segmentation_map = fits.getdata(args.segmentation)
        idx = np.isnan(segmentation_map)
        segmentation_map[idx] = 0

    if args.spines.endswith(".fits"):
        spines = fits.getdata(args.spines)
        idx = np.isnan(spines)
        spines[idx] = 0

    if args.background.endswith(".fits"):
        background = fits.getdata(args.background)
        idx = np.isnan(background)
        background[idx] = 0

    labels = np.full_like(segmentation_map, -1)
    labels[background == 1] = 0
    labels[spines == 1] = 1
    filament_pixels = segmentation_map[spines == 1]
    background_pixels = segmentation_map[background == 1]
    idx = labels >= 0

    dsc_list = []
    band_width = 5000
    band_stride = 1000
    band_start = 0
    while band_start + band_stride <= segmentation_map.shape[1]:
        seg_mask = segmentation_map[:, band_start : band_start + band_width].copy()
        label_mask = labels[:, band_start : band_start + band_width].copy()
        idx = label_mask >= 0
        dsc_list.append(dice_index(seg_mask[idx], label_mask[idx]))
        band_start += band_stride

    dsc_list = np.array(dsc_list)
    plt.plot(np.linspace(360,0,len(dsc_list)), dsc_list, label="Dice for 5000 pixel wide band")
    plt.plot([], [], label=f"Dice mean: {dsc_list.mean()}")
    plt.plot([], [], label=f"Dice var: {dsc_list.var()}")
    plt.xlabel("galactic longitude", fontsize="20")
    plt.ylabel("Dice", fontsize="20")
    plt.title("Random k-fold", fontsize="20")
    plt.legend(fontsize="20")
    plt.show()