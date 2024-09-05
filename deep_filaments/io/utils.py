"""
Utility functions with the OS
"""
import os
import h5py
import numpy as np
import time
import skimage.morphology as skm
import torch
from torch.utils.data import DataLoader
import astropy.io.fits as fits
from skimage.metrics import structural_similarity
from sklearn.metrics import average_precision_score, roc_curve, precision_recall_curve, precision_score, roc_auc_score

from deep_filaments.torch.utils import position_encoding
from deep_filaments.io.dataset import FilamentsDataset
import deep_filaments.utils.informations as dfui
from deep_filaments.torch.train import train_loop, validation_loop
from deep_filaments.torch.models import UNet, UNetPP, SwinUNet, DnCNN, UNet2, UNet_pe, UNet2_pe, UNet_pe_atl, UNet_pe_c
import deep_filaments.torch.models.loss as df_loss
from deep_filaments.metrics.metrics import dice_index
from deep_filaments.metrics.quality_measures import compare_segmentation, get_missed_structures

def flush_into_hdf5(hdf, data, current_index, size, patch_size):
    """
    Flush the current data into the hdf5 file

    Parameters
    ----------
    hdf: h5py.File
        The hdf5 file object
    data: tuple
        The different data (patches, masks/targets, missmap/missing, backgrounds)
    current_index: int
        The current index inside the dataset
    size: int
        The size of the data (current number of patches)
    patch_size:
        The size of the patches
    """
    hdf["patches"].resize((current_index + size, patch_size[0], patch_size[1], 1))
    hdf["patches"][current_index : current_index + size, :, :, :] = data[0][
        :size, :, :, np.newaxis
    ]

    hdf["positions"].resize((current_index + size, 2, 2, 1))
    hdf["positions"][current_index : current_index + size, :, :, :] = data[1][
        :size, :, :, np.newaxis
    ]

    if "spines" in hdf:
        hdf["spines"].resize((current_index + size, patch_size[0], patch_size[1], 1))
        hdf["spines"][current_index : current_index + size, :, :, :] = data[2][
            :size, :, :, np.newaxis
        ]

    if "missing" in hdf:
        hdf["missing"].resize((current_index + size, patch_size[0], patch_size[1], 1))
        hdf["missing"][current_index : current_index + size, :, :, :] = data[3][
            :size, :, :, np.newaxis
        ]

    if "background" in hdf:    
        hdf["background"].resize((current_index + size, patch_size[0], patch_size[1], 1))
        hdf["background"][current_index : current_index + size, :, :, :] = data[4][
            :size, :, :, np.newaxis
        ]
    
    if "normed" in hdf:
        hdf["normed"].resize((current_index + size, patch_size[0], patch_size[1], 1))
        hdf["normed"][current_index : current_index + size, :, :, :] = data[2][
            :size, :, :, np.newaxis
        ]

    hdf.flush()

def extract_patches(
    source,
    output,
    patch_size,
    normed_image=None,
    train=False,
    target=None,
    missing=None,
    background=None,
    train_overlap=0,
    test_overlap=0,
    mask=None,
):
    """
    Extract patches from a source, with a target and a missing data map
    All the data are assumed to be of the same size

    Parameters
    ----------
    source: np.ndarray
        The source data (aka n(h2) map)
    target: np.ndarray
        The target (aka RoI or spines) data
    missing: np.ndarray
        The missing data map (with 1 where data are missing, 0 otherwise)
    background: np.ndarray
        The background data (with 1 for background, 0 otherwise)
    output: str
        The name of the output file (HDF5 file)
    patch_size: tuple[int, int]
        The size of one patch
    overlap: int, optional
        The overlap between 2 patches (0 by default)
    normalizer: callable, optional
        The normalizing function (avoid missing data)
    conservative: bool, optional
        Update the missing data with unlabelled pixels

    Returns
    -------
    A HDF5 reference to the set of patches
    """

    # tmp = np.log10(source[np.nonzero(source)])
    # data_min = np.min(tmp)
    # data_max = np.max(tmp)
    data_min = -1000
    data_max = 1000
    if mask is not None:
        assert mask.shape == source.shape, "The source and the mask have to have the same size"
        train = True
        # Configuration variables
        hdf2_cache = 1000  # The number of patches before a flush
        hdf2_current_size = 0
        hdf2_current_index = 0
        # Initialize the results
        hdf2_data = [
            np.zeros((hdf2_cache, patch_size[0], patch_size[1])),   # Patches
            np.zeros((hdf2_cache, 2, 2)),    # Positions
            np.zeros((hdf2_cache, patch_size[0], patch_size[1])),  # Target
            np.zeros((hdf2_cache, patch_size[0], patch_size[1])),  # Normed
            np.zeros((hdf2_cache, patch_size[0], patch_size[1])),  # Missing
            np.zeros((hdf2_cache, patch_size[0], patch_size[1])),  # Background
            ]
        hdf2_hdf = create_hdf(output + "_test", patch_size, data_min, data_max, True, True, True, True)

    if train:
        assert target is not None, "For creating a training dataset, a target map is required"
        assert missing is not None, "For creating a training dataset, a missing map is required"
        assert background is not None, "For creating a training dataset, a background map is required"

    # Configuration variables
    hdf1_cache = 1000  # The number of patches before a flush
    hdf1_current_size = 0
    hdf1_current_index = 0

    # Initialize the results
    hdf1_data = [
        np.zeros((hdf1_cache, patch_size[0], patch_size[1])),   # Patches
        np.zeros((hdf1_cache, 2, 2))    # Positions
        ]

    if train:
        hdf1_data.append(np.zeros((hdf1_cache, patch_size[0], patch_size[1])))  # Target
        hdf1_data.append(np.zeros((hdf1_cache, patch_size[0], patch_size[1])))  # Normed
        hdf1_data.append(np.zeros((hdf1_cache, patch_size[0], patch_size[1])))  # Missing
        hdf1_data.append(np.zeros((hdf1_cache, patch_size[0], patch_size[1])))  # Background
        hdf1_hdf = create_hdf(output + "_train", patch_size, data_min, data_max, True, True, True, True)
    else:
        hdf1_hdf = create_hdf(output + "_test", patch_size, data_min, data_max)

    # Get patches with corresponding masks and missing data map
    x = 0
    while x <= source.shape[0] - patch_size[0]:
        x_train_bool = x % (patch_size[0] - train_overlap) == 0
        x_test_bool = x % (patch_size[0] - test_overlap) == 0
        if x_train_bool or x_test_bool:
            y = 0
            while y <= source.shape[1] - patch_size[1]:
                y_train_bool = y % (patch_size[1] - train_overlap) == 0
                y_test_bool = y % (patch_size[1] - test_overlap) == 0
                if y_train_bool or y_test_bool:
                    if mask is not None:
                        submask = mask[x : x + patch_size[0], y : y + patch_size[1]]
                        if submask.sum() == patch_size[0] * patch_size[1] and y_test_bool and x_test_bool:
                            hdf2_data, hdf2_current_size = hdf_icrementation(hdf2_data, x, y, patch_size, hdf2_current_size, source, train=True, test=True, normed_image=normed_image, missing=missing, background=background, target=target)
                        if submask.sum() == 0 and y_train_bool and x_train_bool:
                            hdf1_data, hdf1_current_size = hdf_icrementation(hdf1_data, x, y, patch_size, hdf1_current_size, source, train=True, test=False, normed_image=normed_image, missing=missing, background=background, target=target)
                    else:
                        if train and y_train_bool and x_train_bool:
                            hdf1_data, hdf1_current_size = hdf_icrementation(hdf1_data, x, y, patch_size, hdf1_current_size, source, train=True, test=False, normed_image=normed_image, missing=missing, background=background, target=target)
                        elif not train and y_test_bool and x_test_bool:
                            hdf1_data, hdf1_current_size = hdf_icrementation(hdf1_data, x, y, patch_size, hdf1_current_size, source, train=False, test=True, normed_image=normed_image, missing=None, background=None, target=None)

                    # Flush when needed
                    if hdf1_current_size == hdf1_cache:
                        flush_into_hdf5(
                            hdf1_hdf,
                            hdf1_data,
                            hdf1_current_index,
                            hdf1_cache,
                            patch_size,
                        )
                        hdf1_current_index += hdf1_cache
                        hdf1_current_size = 0
                    if mask is not None:
                        # Flush when needed
                        if hdf2_current_size == hdf2_cache:
                            flush_into_hdf5(
                                hdf2_hdf,
                                hdf2_data,
                                hdf2_current_index,
                                hdf2_cache,
                                patch_size,
                            )
                            hdf2_current_index += hdf2_cache
                            hdf2_current_size = 0
                y += 1
        x += 1

    # Final flush
    if hdf1_current_size > 0:
        flush_into_hdf5(
            hdf1_hdf,
            hdf1_data,
            hdf1_current_index,
            hdf1_current_size,
            patch_size,
        )
    # Final flush
    if mask is not None:
        if hdf2_current_size > 0:
            flush_into_hdf5(
                hdf2_hdf,
                hdf2_data,
                hdf2_current_index,
                hdf2_current_size,
                patch_size,
            )
    if mask is not None:
        return [hdf1_hdf, hdf2_hdf]
    else:
        return [hdf1_hdf]

def hdf_icrementation(hdf, x, y, patch_size, hdf_current_size, source, train=False, test=False, normed_image=None, missing=None, background=None, target=None):

    p = source[x : x + patch_size[0], y : y + patch_size[1]]
    position = [[x, x + patch_size[0]], [y, y + patch_size[1]]]
    idx = np.isnan(p)
    p[idx] = 0

    if train:   
        m = missing[x : x + patch_size[0], y : y + patch_size[1]]
        n = normed_image[x : x + patch_size[0], y : y + patch_size[1]]
        b = background[x : x + patch_size[0], y : y + patch_size[1]]
        t = target[x : x + patch_size[0], y : y + patch_size[1]]
        m[idx] = 0

        if test or (not test and np.sum(m) > 1.0):
            hdf[2][hdf_current_size, :, :] = t
            hdf[3][hdf_current_size, :, :] = m
            hdf[5][hdf_current_size, :, :] = n
            hdf[4][hdf_current_size, :, :] = b
            hdf[0][hdf_current_size, :, :] = p
            hdf[1][hdf_current_size, :, :] = position
            hdf_current_size += 1
    else:
        hdf[0][hdf_current_size, :, :] = p
        hdf[1][hdf_current_size, :, :] = position
        hdf_current_size += 1

    return hdf, hdf_current_size

def create_hdf(output, patch_size, data_min, data_max, missing=False, target=False, background=False, normed=False):
    # Creation of the HDF5 file
    hdf = h5py.File(output + ".h5", "w")

    hdf.create_dataset(
        "patches",
        (1, patch_size[0], patch_size[1], 1),
        maxshape=(None, patch_size[0], patch_size[1], 1),
        compression="gzip",
        compression_opts=7,
    )
    hdf.attrs["min"] = data_min
    hdf.attrs["max"] = data_max
    hdf.create_dataset(
        "positions",
        (1, 2, 2, 1),
        maxshape=(None, 2, 2, 1),
        compression="gzip",
        compression_opts=7,
    )
    if missing:
        hdf.create_dataset(
            "missing",
            (1, patch_size[0], patch_size[1], 1),
            maxshape=(None, patch_size[0], patch_size[1], 1),
            compression="gzip",
            compression_opts=7,
        )
    if target:
        hdf.create_dataset(
            "spines",
            (1, patch_size[0], patch_size[1], 1),
            maxshape=(None, patch_size[0], patch_size[1], 1),
            compression="gzip",
            compression_opts=7,
        )
    if background:
        hdf.create_dataset(
            "background",
            (1, patch_size[0], patch_size[1], 1),
            maxshape=(None, patch_size[0], patch_size[1], 1),
            compression="gzip",
            compression_opts=7,
        )
    if normed:
        hdf.create_dataset(
            "normed",
            (1, patch_size[0], patch_size[1], 1),
            maxshape=(None, patch_size[0], patch_size[1], 1),
            compression="gzip",
            compression_opts=7,
        )
    return hdf

def segmentation(dataset_path, model, segmentation_map, count_map, normalization_mode, missing, batch_size, no_segmenter, device):
    dataset = FilamentsDataset(dataset_path, learning_mode="onevsall", data_augmentation=False, normalization_mode=normalization_mode, input_data_noise=0, output_data_noise=0, missmap=missing)
    dataloader = DataLoader(dataset, num_workers=1, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for _, samples in enumerate(dataloader):
            patches = samples["patch"].to(device)
            positions = samples["position"]
            fake_pos = positions.clone()
            fake_pos[:, 0, :, :] = 700
            fake_pos[:, 1, :, :] = 800
            if no_segmenter:
                patch_segmented = patches
            elif isinstance(model, UNet2):
                normed = samples["normed"].to(device)
                patch_segmented = model(torch.cat((patches, normed), dim=1).to(device))
            elif isinstance(model, UNet_pe):
                pe = model.position_encoding(fake_pos).to(device)
                patch_segmented = model(patches, pe)
            elif isinstance(model, UNet2_pe):
                positions = samples["position"]
                normed = samples["normed"].to(device)
                pe = position_encoding(positions).to(device)
                patch_segmented = model(torch.cat((patches, normed), dim=1).to(device), pe)
            elif isinstance(model, UNet_pe_atl):
                pe = model.position_encoding(positions).to(device)
                patch_segmented = model(patches, pe)
            elif isinstance(model, UNet_pe_c):
                pe = model.position_encoding(positions).to(device)
                patch_segmented = model(patches, pe)
            else:
                patch_segmented = model(patches)
            for i in range(patch_segmented.shape[0]):
                segmentation_map[int(positions[i][0][0]) : int(positions[i][0][1]), int(positions[i][1][0]) : int(positions[i][1][1])] += torch.squeeze(patch_segmented[i].cpu().detach()).numpy()
                if missing:
                    missmap = samples["missmap"]
                    segmentation_map[int(positions[i][0][0]) : int(positions[i][0][1]), int(positions[i][1][0]) : int(positions[i][1][1])] *= torch.squeeze(missmap[i]).numpy()
                count_map[int(positions[i][0][0]) : int(positions[i][0][1]), int(positions[i][1][0]) : int(positions[i][1][1])] += + 1
    return segmentation_map, count_map

def segmentation_performances(segmentation, normed_image, spines, background, path, model):
    labels = np.full_like(segmentation, -1)
    labels[background == 1] = 0
    labels[spines == 1] = 1
    filament_pixels = segmentation[spines == 1]
    background_pixels = segmentation[background == 1]
    idx = labels >= 0
    precision, recall, PR_thresholds = precision_recall_curve(labels[idx], segmentation[idx])
    fpr, tpr, ROC_thresholds = roc_curve(labels[idx], segmentation[idx])
    roc_auc = roc_auc_score(labels[idx], segmentation[idx])
    mAP = average_precision_score(labels[idx], segmentation[idx])

    n = 10
    
    threshold_values = np.linspace(0, 1, 100)
    MSSIM_values = np.zeros_like(threshold_values)
    DICE_values = np.zeros_like(threshold_values)
    Precision_values = np.zeros_like(threshold_values)
    F_recovery_values = np.zeros_like(threshold_values)
    B_recovery_values = np.zeros_like(threshold_values)
    for i in range(len(threshold_values)):
        segmentation_binarize_map = segmentation.copy()
        segmentation_binarize_map[segmentation_binarize_map < threshold_values[i]] = 0
        segmentation_binarize_map[segmentation_binarize_map >= threshold_values[i]] = 1
        erode = skm.erosion(segmentation_binarize_map, skm.square(4)) # The value 4 is motivated by Appendix B of https://academic.oup.com/mnras/article/492/4/5420/5731426
        segmentation_binarize_map = skm.reconstruction(erode, segmentation_binarize_map)
        DICE_values[i] = dice_index(segmentation_binarize_map[idx], labels[idx])
        Precision_values[i] = precision_score(segmentation_binarize_map[idx], labels[idx])
        F_recovery_values[i] = compare_segmentation(segmentation, spines, threshold_values[i])
        B_recovery_values[i] = compare_segmentation(1 - segmentation, background, threshold_values[i])
        MSSIM_values[i] = np.mean([structural_similarity(segmentation_binarize_map[: , int(i * segmentation.shape[1] / n) : int((i + 1) * segmentation.shape[1] / n)],
                            normed_image[: , int(i * normed_image.shape[1] / n) : int((i + 1) * normed_image.shape[1] / n)],
                            gaussian_weights=False,
                            use_sample_covariance=False,
                            win_size=31,
                            K1 = 0.00001,
                            K2 = 0.00001,
                            data_range=1)
                            for i in range(n)])
    global_segmentation_binarize_map = segmentation.copy()
    global_threshold = np.argmax(DICE_values) / len(DICE_values)
    global_segmentation_binarize_map[global_segmentation_binarize_map >= global_threshold] = 1
    global_segmentation_binarize_map[global_segmentation_binarize_map < global_threshold] = 0
    erode = skm.erosion(global_segmentation_binarize_map, skm.square(4)) # The value 4 is motivated by Appendix B of https://academic.oup.com/mnras/article/492/4/5420/5731426
    global_segmentation_binarize_map = skm.reconstruction(erode, global_segmentation_binarize_map)
    band_width = 6000
    nb_of_bands = int(spines.shape[1] / band_width)
    local_threshold = np.zeros(nb_of_bands)
    local_local_DSC = np.zeros(nb_of_bands)
    local_segmentation_binarize_map = segmentation.copy()
    for i in range(nb_of_bands):
        label_mask = labels[:,  i * band_width : (i + 1) * band_width]
        local_idx = label_mask >= 0
        mask = segmentation[:,  i * band_width : (i + 1) * band_width].copy()
        for j in range(len(threshold_values)):
            tmp_seg = mask.copy()
            tmp_seg[tmp_seg >= threshold_values[j]] = 1
            tmp_seg[tmp_seg < threshold_values[j]] = 0
            dsc = dice_index(tmp_seg[local_idx], label_mask[local_idx])
            if dsc > local_local_DSC[i]:
                local_threshold[i] = threshold_values[j]
                local_local_DSC[i] = dsc
        mask[mask < local_threshold[i]] = 0
        mask[mask >= local_threshold[i]] = 1
        local_segmentation_binarize_map[:,  i * band_width : (i + 1) * band_width] = mask.copy()
    erode = skm.erosion(local_segmentation_binarize_map, skm.square(4)) # The value 4 is motivated by Appendix B of https://academic.oup.com/mnras/article/492/4/5420/5731426
    local_segmentation_binarize_map = skm.reconstruction(erode, local_segmentation_binarize_map)
    local_DICE = dice_index(local_segmentation_binarize_map[idx], labels[idx])
    local_Precision = precision_score(local_segmentation_binarize_map[idx], labels[idx])
    local_F_recovery = compare_segmentation(local_segmentation_binarize_map, spines, 0.5)
    local_B_recovery = compare_segmentation(1 - local_segmentation_binarize_map, background, 0.5)
    local_MSSIM = np.mean([structural_similarity(local_segmentation_binarize_map[: , int(i * local_segmentation_binarize_map.shape[1] / n) : int((i + 1) * local_segmentation_binarize_map.shape[1] / n)],
                        normed_image[: , int(i * normed_image.shape[1] / n) : int((i + 1) * normed_image.shape[1] / n)],
                        gaussian_weights=False,
                        use_sample_covariance=False,
                        win_size=31,
                        K1 = 0.00001,
                        K2 = 0.00001,
                        data_range=1)
                        for i in range(n)])
    dfui.save_segmentation_performances(path, segmentation, model, filament_pixels, background_pixels, precision, recall, roc_auc, PR_thresholds, fpr, tpr, ROC_thresholds, threshold_values, MSSIM_values, DICE_values, Precision_values, F_recovery_values, B_recovery_values, mAP, local_threshold, local_B_recovery, local_F_recovery, local_DICE, local_MSSIM, local_Precision, local_local_DSC)
    return local_segmentation_binarize_map, global_segmentation_binarize_map

def save_segmentation(local_segmentation_binarize_map, global_segmentation_binarize_map, source, spines, header, folder, prefix):
    seg_density = source.copy()
    seg_density[local_segmentation_binarize_map == 0] = 0
    fits.writeto(os.path.join(folder, prefix + f'_binarize_segmentation_local_threshold.fits'), data=local_segmentation_binarize_map, header=header, overwrite=True)
    fits.writeto(os.path.join(folder, prefix + f'_density_segmentation_local_threshold.fits'), data=seg_density, header=header, overwrite=True)
    new = get_missed_structures(spines, local_segmentation_binarize_map, 0.5)
    new_density = source.copy()
    new_density[new == 0] = 0
    fits.writeto(os.path.join(folder, prefix + f'_new_structures_binarize_local_threshold.fits'), data=new, header=header, overwrite=True)
    fits.writeto(os.path.join(folder, prefix + f'_new_structures_density_local_threshold.fits'), data=new_density, header=header, overwrite=True)

    seg_density = source.copy()
    seg_density[global_segmentation_binarize_map == 0] = 0
    fits.writeto(os.path.join(folder, prefix + f'_binarize_segmentation_global_threshold.fits'), data=global_segmentation_binarize_map, header=header, overwrite=True)
    fits.writeto(os.path.join(folder, prefix + f'_density_segmentation_global_threshold.fits'), data=seg_density, header=header, overwrite=True)
    new = get_missed_structures(spines, global_segmentation_binarize_map, 0.5)
    new_density = source.copy()
    new_density[new == 0] = 0
    fits.writeto(os.path.join(folder, prefix + f'_new_structures_binarize_global_threshold.fits'), data=new, header=header, overwrite=True)
    fits.writeto(os.path.join(folder, prefix + f'_new_structures_density_global_threshold.fits'), data=new_density, header=header, overwrite=True)
    
def early_stopping_check(validation_loss, delta, patience):
    best_loss = np.min(validation_loss)
    tmp = validation_loss[-patience:] - best_loss - delta < 0
    if tmp.sum()==patience:
        return True
    else:
        return False

def train_procedure(
    directory,
    prefix_str,
    model_arch,
    train_path,
    test_path,
    train_batch_size=100,
    test_batch_size=100,
    learning_rate=1e-3,
    epochs=100,
    loss="BCE",
    normalization_mode="none",
    learning_mode="conservative",
    data_augmentation=2,
    input_data_noise=0,
    output_data_noise=0,
    model_to_load=None,
    device="cpu",
):
    """
    Proceed to the training using given environment

    Parameters
    ----------
    data_parameters: DatasetParameters
        The parameters for data management
    train_parameters: TrainParameters
        The parameters for the training process
    model:
        The NN model
    device:
        The device for the computations
    dataloaders: tuple[torch.utils.Dataloader, torch.utils.Dataloader, torch.utils.Dataloader]
        The three dataloader needed for the training
    directories: tuple[Path, Path, Path]
        The three working directory (project, data, models)
    model_to_load: str
        The filename with initial weight (if None we use random initialisation)
    """

    model = None
    if model_arch == "UNetPP":
        model = UNetPP()
    elif model_arch == "SwinUNet":
        model = SwinUNet()
    elif model_arch == "DnCNN":
        model = DnCNN()
    elif model_arch == "UNet":
        model = UNet()
    elif model_arch == "UNet2":
        model = UNet2()
    elif model_arch == "UNet_pe":
        model = UNet_pe()
    elif model_arch == "UNet2_pe":
        model = UNet2_pe()
    elif model_arch == "UNet_pe_atl":
        model = UNet_pe_atl()
    elif model_arch == "UNet_pe_c":
        model = UNet_pe_c()
    model.to(device)
    
    if model_to_load is not None:
        model.state_dict(torch.load(model_to_load, map_location=device))
        model.train()

    train_dataset = FilamentsDataset(dataset_path=train_path, learning_mode=learning_mode, data_augmentation=data_augmentation, normalization_mode=normalization_mode, input_data_noise=input_data_noise, output_data_noise=output_data_noise, missmap=False)
    
    early_stopping = False
    if early_stopping:
        n = len(train_dataset)
        nn = int(n * 0.9)
        train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset, [nn, n - nn])
        validation_dataloader = DataLoader(validation_dataset, num_workers=1, batch_size=train_batch_size, shuffle=True)

    train_dataloader = DataLoader(train_dataset, num_workers=1, batch_size=train_batch_size, shuffle=True)

    if loss == "CE":
        loss_fn = torch.nn.CrossEntropyLoss()
    elif loss == "MSE":
        loss_fn = torch.nn.MSELoss()
    elif loss == "BCE":
        loss_fn = torch.nn.BCELoss()  # Very good
    elif loss == "L1":
        loss_fn = torch.nn.L1Loss()
    elif loss == "DICE":
        loss_fn = df_loss.DiceLoss()  # Very good
    elif loss == "BCEDICE":
        loss_fn = df_loss.BinaryCrossEntropyDiceSum()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # The scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, [10, 20, 30, 40, 50, 60, 70, 80, 90], gamma=0.1
    )

    train_loss = np.zeros(epochs)
    val_loss = np.zeros(epochs)
    # register dice values for 4 thresholds
    train_dice = np.zeros((epochs, 4))
    val_dice = np.zeros((epochs, 4))

    timestamp = int(time.time())

    t = 0
    while t < epochs: # and True if t < 5 else early_stopping_check(val_loss[: t], 0.00001, 5):
        print(f"Epoch {t + 1}\n-------------------------------")
        epoch_train_loss, epoch_train_dice = train_loop(train_dataloader, model, loss_fn, optimizer, device)
        # epoch_val_loss, epoch_val_dice = validation_loop(validation_dataloader, model, loss_fn, device, name="Validation")

        lr_scheduler.step()
        train_loss[t] = epoch_train_loss
        train_dice[t] = epoch_train_dice
        # val_loss[t] = epoch_val_loss
        # val_dice[t] = epoch_val_dice
        t += 1

    torch.save(model.state_dict(), os.path.join(directory, prefix_str + "_final_model.pt"))

    test_dataset = FilamentsDataset(dataset_path=test_path, learning_mode=learning_mode, data_augmentation=0, normalization_mode=normalization_mode, input_data_noise=0, output_data_noise=0, missmap=False)
    n = len(test_dataset)
    nn = int(n * 0.005)
    test_dataset, _ = torch.utils.data.random_split(test_dataset, [nn, n - nn])
    test_dataloader = DataLoader(test_dataset, num_workers=1, batch_size=test_batch_size, shuffle=True)
    test_loss, test_dice = validation_loop(test_dataloader, model, loss_fn, device, name="Test")

    metrics = {
        "train_loss": train_loss,
        "train_dice": train_dice,
        "test_loss": test_loss,
        "test_dice": test_dice,
        "val_loss": val_loss,
        "val_dice": val_dice,
    }

    dfui.save_experiment_info(
        path=str(os.path.join(directory,prefix_str +"_training.hdf5")),
        metrics=metrics,
        timestamp=timestamp,
        loss_fn=loss_fn,
        epochs=epochs,
        batch_size=train_batch_size,
        learning_rate=learning_rate,
        data_augmentation=data_augmentation,
        input_data_noise=input_data_noise,
        output_data_noise=output_data_noise
    )

def get_sorted_file_list(directory):
    """
    Get a sorted list a files

    Parameters
    ----------
    directory: str
        The involved directory with the files

    Returns
    -------
    A sorted list of file
    """
    pwd = os.scandir(directory)
    files = []
    for f in pwd:
        if f.name.endswith(".fits") or f.name.endswith(".fit") or f.name.endswith(".fits.gz") or f.name.endswith(".h5") or f.name.endswith(".npy") or f.name.endswith(".pt"):
            files.append(f.name)
    files.sort()
    return files

def get_mosaic_angles(file: str) -> str:
    """
    Get the angles of the mosaic

    Parameters
    ----------
    file: str
        The filename of the mosaic image

    Returns
    -------
    The corresponding mosaic angle
    """
    angles = file.split("_")[1][6:12]
    return angles

def kfold_statistics(path, train_dataset_path, test_dataset_path, batch_size):
    train_dataset_files = get_sorted_file_list(train_dataset_path)
    test_dataset_files = get_sorted_file_list(test_dataset_path)

    number_of_test_patches = np.zeros(len(train_dataset_files))
    number_of_train_patches = number_of_test_patches.copy()
    number_of_train_spine_pixels = number_of_test_patches.copy()
    number_of_train_missing_pixels = number_of_test_patches.copy()
    number_of_test_missing_pixels = number_of_test_patches.copy()
    number_of_test_spine_pixels = number_of_test_patches.copy()
    number_of_train_background_pixels = number_of_test_patches.copy()
    number_of_test_background_pixels = number_of_test_patches.copy()

    for i in range(len(train_dataset_files)):
        dataset = FilamentsDataset(os.path.join(test_dataset_path, test_dataset_files[i]), learning_mode="conservative", data_augmentation=False, input_data_noise=0, output_data_noise=0, missmap=False)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for _, samples in enumerate(dataloader):
            number_of_test_patches[i] += samples["patch"].shape[0]
            number_of_test_spine_pixels[i] += torch.sum(samples["target"])
            number_of_test_background_pixels[i] += torch.sum(samples["background"])
            number_of_test_missing_pixels[i] += torch.sum(samples["missing"])
        
        dataset = FilamentsDataset(os.path.join(train_dataset_path, train_dataset_files[i]), learning_mode="conservative", data_augmentation=False, input_data_noise=0, output_data_noise=0, missmap=False)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for _, samples in enumerate(dataloader):
            number_of_train_patches[i] += samples["patch"].shape[0]
            number_of_train_spine_pixels[i] += torch.sum(samples["target"])
            number_of_train_background_pixels[i] += torch.sum(samples["background"])
            number_of_train_missing_pixels[i] += torch.sum(samples["missing"])

    dfui.save_kfold_statistics(path, number_of_train_patches, number_of_test_patches, number_of_train_spine_pixels, number_of_train_background_pixels, number_of_test_spine_pixels, number_of_test_background_pixels, number_of_train_missing_pixels, number_of_test_missing_pixels)