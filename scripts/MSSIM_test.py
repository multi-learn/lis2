import argparse

from multiprocessing import Pool
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import numpy as np
from skimage.metrics import structural_similarity
import skimage.morphology as morpho


def extract_filaments(segmentation_map):
    filaments = morpho.label(segmentation_map, connectivity=2)

    xmin = np.zeros(np.max(filaments) - 1)
    xmax = np.zeros(np.max(filaments) - 1)
    ymin = np.zeros(np.max(filaments) - 1)
    ymax = np.zeros(np.max(filaments) - 1)
    
    for k in range(1, np.max(filaments)):
        x = []
        y = []
        for i in range(filaments.shape[0]):
            for j in range(filaments.shape[1]):
                if filaments[i,j] == k:
                    x.append(i)
                    y.append(j)
        xmin[k - 1] = np.min(x)
        xmax[k - 1] = np.max(x) + 1
        ymin[k - 1] = np.min(y)
        ymax[k - 1] = np.max(y) + 1

    return filaments, xmin, xmax, ymin, ymax

def false_filament_test_loop(segmentation_map, normed_image, filaments, xmin, xmax, ymin, ymax, overlap, idx_fil, ssim_mode, win_size):
    count = 0
    filaments_count = 0
    heat_map = np.zeros_like(segmentation_map)
    full_mask = filaments.copy()
    full_mask[full_mask != idx_fil + 1] = 0
    mask = full_mask[int(xmin[idx_fil]) : int(xmax[idx_fil]), int(ymin[idx_fil]) : int(ymax[idx_fil])]
    x_size = int(xmax[idx_fil] - xmin[idx_fil])
    y_size = int(ymax[idx_fil] - ymin[idx_fil])
    for i in range(overlap, segmentation_map.shape[0] - overlap):
        for j in range(overlap, segmentation_map.shape[1] - overlap):
            filaments_count_res, count_res = place_false_filament_loop(segmentation_map, normed_image, mask, overlap, x_size, y_size, i, j, ssim_mode, win_size)
            count += count_res
            filaments_count += filaments_count_res
            if filaments_count_res == 1:
                heat_map[i : i + x_size, j : j + y_size] = mask
    return count, filaments_count, heat_map

def false_filament_test(segmentation_map, normed_image, filaments, xmin, xmax, ymin, ymax, ssim_mode, win_size):
    overlap = 16
    with Pool() as p:
        results = [p.apply_async(false_filament_test_loop, args=(segmentation_map, normed_image, filaments, xmin, xmax, ymin, ymax, overlap, idx_fil, ssim_mode, win_size)) for idx_fil in range(len(xmin))]
        count = np.array([res.get()[0] for res in results]).sum()
        filaments_count = np.array([res.get()[1] for res in results]).sum()
        heat_map = np.array([res.get()[2] for res in results]).sum(axis=0)
    return filaments_count / count, heat_map

def place_false_filament_loop(segmentation_map, normed_image, mask, overlap, x_size, y_size, i, j, ssim_mode, win_size):
    if j + y_size <= segmentation_map.shape[1] and i + x_size <= segmentation_map.shape[0] and segmentation_map[i - 1 : i + x_size + 1, j - 1 : j + y_size + 1].sum() == 0:
        tmp = segmentation_map.copy()
        tmp[i : i + x_size, j : j + y_size] = mask
        tmp_mssim = compute_mssim(tmp[i - overlap : i + x_size + overlap, j - overlap : j + y_size + overlap],
                                normed_image[i - overlap : i + x_size + overlap, j - overlap : j + y_size + overlap], 
                                ssim_mode, win_size)
        original_mssim = compute_mssim(segmentation_map[i - overlap : i + x_size + overlap, j - overlap : j + y_size + overlap],
                                normed_image[i - overlap : i + x_size + overlap, j - overlap : j + y_size + overlap], 
                                ssim_mode, win_size)
        if tmp_mssim > original_mssim:
            return 1, 1
        else:
            return 0, 1
    return 0, 0

def compute_mssim(segmentatation, normed, mode, win_size=7):
    if mode == "gaussian":
        return structural_similarity(segmentatation,
                                normed, 
                                gaussian_weights=True,
                                use_sample_covariance=False,
                                sigma=1.5,
                                K1 = 0.00001,
                                K2 = 0.00001,
                                data_range=1)
    elif mode == "block":
        return structural_similarity(segmentatation,
                                normed, 
                                gaussian_weights=False,
                                win_size=win_size,
                                use_sample_covariance=False,
                                sigma=1.5,
                                K1 = 0.00001,
                                K2 = 0.00001,
                                data_range=1)
    
def delete_filament_test_loop(segmentation_map, normed_image, filaments, ssim_mode, win_size):
    nb_of_filament_del = 0
    mssim = compute_mssim(segmentation_map,
                            normed_image, 
                            ssim_mode, 
                            win_size)

    for k in range(1, np.max(filaments)):
        tmp = segmentation_map.copy()
        tmp[filaments == k] = 0
        mssim_2 = compute_mssim(tmp,
                                normed_image, 
                                ssim_mode, 
                                win_size)
        if mssim_2 > mssim:
            nb_of_filament_del += 1
    return nb_of_filament_del

def mssim_test_loop(segmentation_map, normed_image, filaments, xmin, xmax, ymin, ymax, ssim_mode, win_size):
    pourcentage_of_false_filament, heat_map = false_filament_test(segmentation_map, normed_image, filaments, xmin, xmax, ymin, ymax, ssim_mode, win_size)
    nb_of_del_filament = delete_filament_test_loop(segmentation_map, normed_image, filaments, ssim_mode, win_size)
    return pourcentage_of_false_filament, nb_of_del_filament, heat_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Measure the quality of a segmentation."
    )
    parser.add_argument("segmentation", help="The segmentation result", type=str)
    parser.add_argument("normed_image", help="Normed density image", type=str)
    parser.add_argument("--ssim_mode", help="Way to compute the mssim. Has to be either guassian or block", type=str, default="gaussian"
    )
    parser.add_argument("--win_size", help="size of the window to compute the mssim. Has to be odd", type=int, default=7
    )
    args = parser.parse_args()

    segmentation_map = fits.getdata(args.segmentation)
    idx = np.isnan(segmentation_map)
    segmentation_map[idx] = 0
    header = fits.getheader(args.segmentation)

    normed_image = fits.getdata(args.normed_image)
    idx = np.isnan(normed_image)
    normed_image[idx] = 0

    # normed_image = np.log(normed_image)
    # normed_image -= normed_image.min()
    # normed_image /= normed_image.max()

    filaments, xmin, xmax, ymin, ymax = extract_filaments(segmentation_map)

    filament_pourcent, nb_of_filament_del, heat_map = mssim_test_loop(segmentation_map, normed_image, filaments, xmin, xmax, ymin, ymax, args.ssim_mode, args.win_size)

    print(f"Number of filaments deleted: {nb_of_filament_del} -- pourcentage of filament added: {filament_pourcent}")
    plt.imshow(heat_map)
    plt.colorbar()
    plt.show()