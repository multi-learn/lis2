"""
Function for saving information during training and others
"""
import inspect

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.rcParams.update({'font.size': 20})

def save_experiment_info(
    path, metrics, timestamp, loss_fn, epochs, batch_size, learning_rate, data_augmentation, input_data_noise, output_data_noise
):
    """Save experiment information in the hdf5 format.

    Each experiment is identified by its timestamp.
    The following attrs are assigned for an experiment in the hdf5 file:
        * architecture
        * rnd_seed
        * timestamp
        * epochs
        * batch_size
        * learning_rate
        * output
        * patches
        * percent

    The following metrics are attached to each experiment:
        * train_loss
        * test_loss
        * train_dice
        * test_dice

    The train_loss, test_loss datasets also have "name" and "module" attributes
    to identify loss functions.

    Parameters
    ----------
    path : str
        Location of the hdf5 file.
    data_parameters: DataParameters
        The parameters for managing the data
    train_parameters: TrainParameters
        The parameters for the training process
    metrics : dict
        Named collection of metrics to be saved, i.e. "train_loss", "test_loss",
        "train_dice", "test_dice"
    timestamp : int
        A date when the experiment was conducted measured in seconds.
    loss_fn : torch.nn.Module
        The current loss function.
    """
    with h5py.File(path, "a") as f:
        new_exp = f.create_group(f"exp_{timestamp}")

        loss_keys = ("train_loss", "val_loss", "test_loss")

        for key in loss_keys:
            loss = metrics[key]
            loss_dataset = new_exp.create_dataset(key, data=loss)
            loss_dataset.attrs["name"] = repr(loss_fn)
            loss_dataset.attrs["module"] = repr(inspect.getmodule(loss_fn))

        dice_keys = ("train_dice", "val_dice", "test_dice")

        for key in dice_keys:
            dice = metrics[key]
            dice = new_exp.create_dataset(key, data=dice)

        new_exp.attrs["architecture"] = "UNet"
        new_exp.attrs["timestamp"] = timestamp
        new_exp.attrs["epochs"] = epochs
        new_exp.attrs["batch_size"] = batch_size
        new_exp.attrs["learning_rate"] = learning_rate
        new_exp.attrs["data_augmentation"] = data_augmentation
        new_exp.attrs["input_data_noise"] = input_data_noise
        new_exp.attrs["output_data_noise"] = output_data_noise

def save_segmentation_performances(path, segmentation, model, filament_pixels, background_pixels, precision, recall, roc_auc, PR_thresholds, fpr, tpr, ROC_thresholds, threshold_values, MSSIM_values, DICE_values, Precision_values, F_recovery_values, B_recovery_values, mAP, local_threshold, local_B_recovery, local_F_recovery, local_DICE, local_MSSIM, local_Precision, local_local_DSC):
    with h5py.File(path, "w") as seg_perf:
        seg_perf.create_dataset("segmentation", data=segmentation)
        seg_perf.create_dataset("model", data=[model])
        seg_perf.create_dataset("filament_pixels", data=filament_pixels)
        seg_perf.create_dataset("background_pixels", data=background_pixels)
        seg_perf.create_dataset("precision", data=precision)
        seg_perf.create_dataset("recall", data=recall)
        seg_perf.create_dataset("PR_thresholds", data=PR_thresholds)
        seg_perf.create_dataset("fpr", data=fpr)
        seg_perf.create_dataset("roc_auc", data=roc_auc)
        seg_perf.create_dataset("tpr", data=tpr)
        seg_perf.create_dataset("ROC_thresholds", data=ROC_thresholds)
        seg_perf.create_dataset("threshold_values", data=threshold_values)
        seg_perf.create_dataset("MSSIM_values", data=MSSIM_values)
        seg_perf.create_dataset("DICE_values", data=DICE_values)
        seg_perf.create_dataset("Precision_values", data=Precision_values)
        seg_perf.create_dataset("F_recovery_values", data=F_recovery_values)
        seg_perf.create_dataset("B_recovery_values", data=B_recovery_values)
        seg_perf.create_dataset("mAP", data=mAP)
        seg_perf.create_dataset("local_threshold_values", data=local_threshold)
        seg_perf.create_dataset("local_local_DSC", data=local_local_DSC)
        seg_perf.create_dataset("local_MSSIM_values", data=local_MSSIM)
        seg_perf.create_dataset("local_DICE_values", data=local_DICE)
        seg_perf.create_dataset("local_Precision_values", data=local_Precision)
        seg_perf.create_dataset("local_F_recovery_values", data=local_F_recovery)
        seg_perf.create_dataset("local_B_recovery_values", data=local_B_recovery)

def save_kfold_statistics(path, number_of_train_patches, number_of_test_patches, number_of_train_spine_pixels, number_of_train_background_pixels, number_of_test_spine_pixels, number_of_test_background_pixels, number_of_train_missing_pixels, number_of_test_missing_pixels):
    with h5py.File(path, "w") as seg_perf:
        seg_perf.create_dataset("number_of_train_patches", data=number_of_train_patches)
        seg_perf.create_dataset("number_of_test_patches", data=number_of_test_patches)
        seg_perf.create_dataset("number_of_train_spine_pixels", data=number_of_train_spine_pixels)
        seg_perf.create_dataset("number_of_train_background_pixels", data=number_of_train_background_pixels)
        seg_perf.create_dataset("number_of_test_spine_pixels", data=number_of_test_spine_pixels)
        seg_perf.create_dataset("number_of_test_background_pixels", data=number_of_test_background_pixels)
        seg_perf.create_dataset("number_of_train_missing_pixels", data=number_of_train_missing_pixels)
        seg_perf.create_dataset("number_of_test_missing_pixels", data=number_of_test_missing_pixels)

def read_kfold_statistics(path):
    with h5py.File(path, "r") as seg_perf:
        number_of_train_patches = np.array(seg_perf.get("number_of_train_patches"))
        number_of_test_patches = np.array(seg_perf.get("number_of_test_patches"))
        number_of_train_spine_pixels = np.array(seg_perf.get("number_of_train_spine_pixels"))
        number_of_train_background_pixels = np.array(seg_perf.get("number_of_train_background_pixels"))
        number_of_test_spine_pixels = np.array(seg_perf.get("number_of_test_spine_pixels"))
        number_of_test_background_pixels = np.array(seg_perf.get("number_of_test_background_pixels"))
        number_of_test_missing_pixels = np.array(seg_perf.get("number_of_test_missing_pixels"))
        number_of_train_missing_pixels = np.array(seg_perf.get("number_of_train_missing_pixels"))

    k = number_of_train_patches.shape[0]

    G = gridspec.GridSpec(2, 4)
    axes_1 = plt.subplot(G[0, 0])
    axes_1.plot(np.linspace(0,k - 1, k), number_of_train_patches, label="train", marker="+")
    axes_1.set_ylabel("number of patches")
    axes_1.set_title("Number of patch in each fold")
    axes_1.legend()

    axes_5 = plt.subplot(G[1, 0])
    axes_5.plot(np.linspace(0,k - 1, k), number_of_test_patches, label="test", marker="*")
    axes_5.set_ylabel("number of patches")
    axes_5.set_title("Number of patch in each fold")
    axes_5.legend()

    axes_2 = plt.subplot(G[0, 1])
    axes_2.plot(np.linspace(0,k - 1, k), number_of_train_spine_pixels, label="train", marker="+")
    axes_2.set_ylabel("number of patches")
    axes_2.set_title("Number of filament pixels in each fold")
    axes_2.legend()

    axes_6 = plt.subplot(G[1, 1])
    axes_6.plot(np.linspace(0,k - 1, k), number_of_test_spine_pixels, label="test", marker="*")
    axes_6.set_ylabel("number of patches")
    axes_6.set_title("Number of filament pixels in each fold")
    axes_6.legend()

    axes_3 = plt.subplot(G[0, 2])
    axes_3.plot(np.linspace(0,k - 1, k), number_of_train_background_pixels, label="train" ,marker="+")
    axes_3.set_xlabel("fold")
    axes_3.set_ylabel("number of patches")
    axes_3.set_title("Number of background pixels in each fold")
    axes_3.legend()

    axes_7 = plt.subplot(G[1, 2])
    axes_7.plot(np.linspace(0,k - 1, k), number_of_test_background_pixels, label="test", marker="*")
    axes_7.set_xlabel("fold")
    axes_7.set_ylabel("number of patches")
    axes_7.set_title("Number of background pixels in each fold")
    axes_7.legend()

    axes_4 = plt.subplot(G[0, 3])
    axes_4.plot(np.linspace(0,k - 1, k), number_of_train_missing_pixels, label="train" ,marker="+")
    axes_4.set_xlabel("fold")
    axes_4.set_ylabel("number of patches")
    axes_4.set_title("Number of missing pixels in each fold")
    axes_4.legend()

    axes_8 = plt.subplot(G[1, 3])
    axes_8.plot(np.linspace(0,k - 1, k), number_of_test_missing_pixels, label="test", marker="*")
    axes_8.set_xlabel("fold")
    axes_8.set_ylabel("number of patches")
    axes_8.set_title("Number of missing pixels in each fold")
    axes_8.legend()

    plt.show()

def read_segmentation_perf(input_file):
    with h5py.File(input_file, "r") as seg_perf:
        segmentation = np.array(seg_perf.get("segmentation"))
        model = str(seg_perf.get("model")[0])
        filament_pixels = np.array(seg_perf.get("filament_pixels"))
        background_pixels = np.array(seg_perf.get("background_pixels"))
        precision = np.array(seg_perf.get("precision"))
        recall = np.array(seg_perf.get("recall"))
        PR_thresholds = np.array(seg_perf.get("PR_thresholds"))
        fpr = np.array(seg_perf.get("fpr"))
        roc_auc = np.array(seg_perf.get("roc_auc"))
        tpr = np.array(seg_perf.get("tpr"))
        ROC_thresholds = np.array(seg_perf.get("ROC_thresholds"))
        threshold_values = np.array(seg_perf.get("threshold_values"))
        MSSIM_values = np.array(seg_perf.get("MSSIM_values"))
        DICE_values = np.array(seg_perf.get("DICE_values"))
        Precision_values = np.array(seg_perf.get("Precision_values"))
        F_recovery_values = np.array(seg_perf.get("F_recovery_values"))
        B_recovery_values = np.array(seg_perf.get("B_recovery_values"))
        mAP = np.array(seg_perf.get("mAP"))
        local_threshold = np.array(seg_perf.get("local_threshold_values"))
        local_local_DSC = np.array(seg_perf.get("local_local_DSC"))
        local_MSSIM = np.array(seg_perf.get("local_MSSIM_values"))
        local_DICE = np.array(seg_perf.get("local_DICE_values"))
        local_Precision = np.array(seg_perf.get("local_Precision_values"))
        local_F_recovery = np.array(seg_perf.get("local_F_recovery_values"))
        local_B_recovery = np.array(seg_perf.get("local_B_recovery_values"))

    print(f"Local threshold to maximize DSC, \nMSSIM: {local_MSSIM}\nDICE: {local_DICE}\nPrecision: {local_Precision}\nFilament recovery: {local_F_recovery}\nBackground recovery: {local_B_recovery}\nMean local threshold: {np.mean(local_threshold)}\nLocal threshold var: {np.var(local_threshold)}\nLocal DSC var: {np.var(local_local_DSC)}")
    print(local_threshold)
    print(local_local_DSC)
    DICE_argmax = np.argmax(DICE_values) / len(DICE_values)
    MSSIM_argmax = np.argmax(MSSIM_values) / len(MSSIM_values)

    G = gridspec.GridSpec(2, 2)
    plt.suptitle(model)
    axes_1 = plt.subplot(G[0, :])
    axes_1.plot(threshold_values, DICE_values, label="DICE")
    axes_1.plot(threshold_values, MSSIM_values, label="MSSIM")
    axes_1.plot(threshold_values, F_recovery_values, label="Filament recovery")
    axes_1.plot(threshold_values, Precision_values, label="Filament precision")
    axes_1.plot(threshold_values, B_recovery_values, label="Background recovery")
    axes_1.vlines(DICE_argmax, 0, np.max(DICE_values), colors='r', linestyles='dashed', label=f'Optimal DICE threshold: {DICE_argmax}')
    axes_1.vlines(MSSIM_argmax, 0, np.max(MSSIM_values), colors='y', linestyles='dashed', label=f'Optimal MSSIM threshold: {MSSIM_argmax}')
    axes_1.plot([0], [0], label=f"mAP = {mAP}", c='w')
    axes_1.plot([0], [0], label=f"Mean DICE = {np.mean(DICE_values)}", c='w')
    axes_1.plot([0], [0], label=f"Optimal DICE = {np.max(DICE_values)}", c='w')
    axes_1.plot([0], [0], label=f"Optimal MSSIM = {np.max(MSSIM_values)}", c='w')
    axes_1.plot([0], [0], label=f"AUC ROC = {roc_auc}", c='w')
    axes_1.set_xlabel('Segmentation threshold')
    axes_1.set_ylabel('Metric values')
    axes_1.set_title('Metric against segmentation threshold analysis')
    axes_1.legend()

    axes_2 = plt.subplot(G[1, 0])
    axes_2.plot(recall[:-1], PR_thresholds, label="Segmentation threshold recall curve")
    axes_2.plot(recall, precision, label="Precision recall curve")
    axes_2.set_title("Filament segmentation: Precision-Recall curve")
    axes_2.set_xlabel("Recall")
    axes_2.set_ylabel("Precision/Segmentation threshold")
    axes_2.legend()

    axes_3 = plt.subplot(G[1, 1])
    axes_3.plot(ROC_thresholds[1:], tpr[1:], label="Segmentation threshold against True Positive rate curve")
    axes_3.plot(fpr, tpr, label="ROC curve")
    axes_3.set_title("Filament segmentation: Receiver operating characteristic curve")
    axes_3.set_xlabel("True Positive Rate")
    axes_3.set_ylabel("False Positive Rate/Segmentation threshold")
    axes_3.legend()
    plt.show()

    G = gridspec.GridSpec(2, 2)
    print(model)
    plt.suptitle(f"{model} : segmentation value distributions")
    axes_1 = plt.subplot(G[0, :])
    axes_1.hist([filament_pixels, background_pixels], bins=np.linspace(0,1,100), label=["Filament", "Background"])
    axes_1.set_xlabel("segmentation value")
    axes_1.set_ylabel("pixel count")
    axes_1.set_title("Segmentation value distributions for pixels with known labels")
    axes_1.legend()

    tmp = segmentation[segmentation > 0]
    axes_2 = plt.subplot(G[1, :])
    axes_2.hist(tmp, bins=np.linspace(0,1,100))
    axes_2.set_xlabel("segmentation value")
    axes_2.set_ylabel("pixel count")
    axes_2.set_title("Segmentation value distributions over the whole image")
    plt.show()