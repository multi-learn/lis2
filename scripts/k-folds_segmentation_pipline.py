"""
Creation of dataset from given n(h2), RoI and mask (missing data)
"""
import argparse
import os
import sys

import numpy as np
import astropy.io.fits as fits
import torch

import deep_filaments.io.utils as utils
from deep_filaments.torch.models import UNet, UNetPP, SwinUNet, DnCNN, UNet2, UNet_pe, UNet2_pe, UNet_pe_atl, UNet_pe_c
from deep_filaments.utils.options import TrainParameters, DatasetParameters

if __name__ == "__main__":
    p_train_parameters = TrainParameters()
    p_data_parameters = DatasetParameters()
    parser = argparse.ArgumentParser(description="Patch dataset construction.")
    parser.add_argument("image", help="The n(h2) image", type=str)
    parser.add_argument("normed_image", help="The normed n(h2) image", type=str)
    parser.add_argument("roi", help="The corresponding RoI", type=str)
    parser.add_argument("missing", help="The missing data mask", type=str)
    parser.add_argument("background", help="The background data", type=str)
    parser.add_argument("test_dataset_path", help="The path to the hdf5 test dataset ", type=str)
    parser.add_argument("train_dataset_path", help="The path to the hdf5 train dataset ", type=str)
    parser.add_argument("--folder", help="The output database file", type=str, default="k_fold"
    )
    parser.add_argument(
        "--train_batch_size",
        help="The size of the batch",
        type=int,
        default=p_data_parameters.batch_size,
    )
    parser.add_argument(
        "--test_batch_size",
        help="The size of the batch",
        type=int,
        default=p_data_parameters.batch_size,
    )
    parser.add_argument(
        "-d",
        "--data_augmentation",
        help="Use data augmentation",
        type=int,
        default=p_data_parameters.data_augmentation,
    )
    parser.add_argument(
        "-e",
        "--epochs",
        help="The number of epochs",
        type=int,
        default=p_train_parameters.epochs,
    )
    parser.add_argument(
        "-l", "--load_model", help="Load an existing model", type=str, default=None
    )
    parser.add_argument(
        "--lr",
        help="Learning rate",
        type=float,
        default=p_train_parameters.learning_rate,
    )
    parser.add_argument(
        "--input_data_noise",
        help="The sigma of the Gaussian noise inside the data augmentation (input)",
        type=float,
        default=p_data_parameters.input_data_noise,
    )
    parser.add_argument(
        "--output_data_noise",
        help="The sigma of the Gaussian noise inside the data augmentation (output)",
        type=float,
        default=p_data_parameters.output_data_noise,
    )
    parser.add_argument(
        "--learning_mode",
        help="Use the background/RoI only",
        choices=["oneclass", "onevsall", "conservative"],
        default="onevsall",
        type=str,
    )
    parser.add_argument(
        "--file_prefix",
        help="Set the file profile (default: ML_segmentation)",
        type=str,
        default="ML_segmentation",
    )
    parser.add_argument(
        "--normalization_mode",
        help="Normalize patches mode for training",
        type=str,
        default="none",
        choices=["direct", "log10", "none"],
    )
    parser.add_argument(
        "--loss",
        help="loss for training",
        type=str,
        default="BCE",
        choices=["BCE", "CE", "L1", "DICE", "MSE", "BCEDICE"],
    )
    parser.add_argument(
        "--model",
        help="The network model",
        default="UNet",
        type=str,
        choices=["UNet", "UNetPP", "SwinUNet", "DnCNN", "UNet2", "UNet_pe", "UNet2_pe", "UNet_pe_atl", "UNet_pe_c"],
    )
    parser.add_argument(
        "--stats",
        help="Compute k-fold statistics",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--performances",
        help="Compute segmentation performances",
        action="store_true",
        default=False,
    )
    
    args = parser.parse_args()

    train_dataset_files = utils.get_sorted_file_list(args.train_dataset_path)
    test_dataset_files = utils.get_sorted_file_list(args.test_dataset_path)

    os.system(f"mkdir {args.folder}")
    os.system(f"mkdir {os.path.join(args.folder, 'models')}")

    
    if args.stats:
        utils.kfold_statistics(os.path.join(args.folder, "kfold_statistics.h5"), args.train_dataset_path, args.test_dataset_path, args.test_batch_size)

    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    
    for i in range(len(train_dataset_files)):
        print("*" * 20)
        print(f"Fold {i}")
        print("*" * 20)
        utils.train_procedure(os.path.join(args.folder, "models"),
                        args.file_prefix + f"_fold_{i}",
                        args.model,
                        os.path.join(args.train_dataset_path, train_dataset_files[i]),
                        os.path.join(args.test_dataset_path, test_dataset_files[i]),
                        train_batch_size=args.train_batch_size,
                        test_batch_size=args.test_batch_size,
                        learning_rate=args.lr,
                        epochs=args.epochs,
                        loss=args.loss,
                        normalization_mode=args.normalization_mode,
                        learning_mode=args.learning_mode,
                        data_augmentation=args.data_augmentation,
                        input_data_noise=args.input_data_noise,
                        output_data_noise=args.output_data_noise,
                        model_to_load=args.load_model,
                        device=device
    )   

    torch.cuda.empty_cache()

    source_data = fits.getdata(args.image)
    source_header = fits.getheader(args.image)
    segmentation_map = np.zeros_like(source_data)
    count_map = segmentation_map.copy()

    model_files = utils.get_sorted_file_list(os.path.join(args.folder, "models"))

    for i in range(len(test_dataset_files)):
        if model_files[i].endswith(".pt"):
            model = None
            if args.model == "UNetPP":
                model = UNetPP()
            elif args.model == "SwinUNet":
                model = SwinUNet()
            elif args.model == "DnCNN":
                model = DnCNN()
            elif args.model == "UNet":
                model = UNet()
            elif args.model == "UNet2":
                model = UNet2()
            elif args.model == "UNet_pe":
                model = UNet_pe()
            elif args.model == "UNet2_pe":
                model = UNet2_pe()
            elif args.model == "UNet_pe_atl":
                model = UNet_pe_atl()
            elif args.model == "UNet_pe_c":
                model = UNet_pe_c()
            else:
                sys.exit(1)
            model.load_state_dict(
                torch.load(
                    os.path.join(args.folder, "models", model_files[i]),
                )
            )
            model.eval()
            model.to(device)
        else:
            sys.exit(1)

        segmentation_map, count_map = utils.segmentation(os.path.join(args.test_dataset_path, test_dataset_files[i]), model, segmentation_map, count_map, args.normalization_mode, True, args.test_batch_size, False, device)
    idx = count_map > 0
    segmentation_map[idx] = segmentation_map[idx] / count_map[idx]
    fits.writeto(os.path.join(args.folder, args.file_prefix + "_merged_segmentation.fits"), data=segmentation_map, header=source_header, overwrite=True)

    if args.performances:
        if args.image.endswith(".fits"):
            img = fits.getdata(args.image)
            idx = np.isnan(img)
            img[idx] = 0
            header = fits.getheader(args.image)

        if args.normed_image.endswith(".fits"):
            normed_image = fits.getdata(args.normed_image)
            idx = np.isnan(normed_image)
            normed_image[idx] = 0

        if args.roi.endswith(".fits"):
            roi = fits.getdata(args.roi)
            idx = np.isnan(roi)
            roi[idx] = 0

        if args.missing.endswith(".fits"):
            missing = fits.getdata(args.missing)
            idx = np.isnan(missing)
            missing[idx] = 0

        if args.background.endswith(".fits"):
            background = fits.getdata(args.background)
            idx = np.isnan(background)
            background[idx] = 0

        local_segmentation_binarize_map, global_segmentation_binarize_map = utils.segmentation_performances(segmentation_map, normed_image, roi, background, os.path.join(args.folder, args.file_prefix + "_segmenation_performances.hdf5"), args.model)
        utils.save_segmentation(global_segmentation_binarize_map, local_segmentation_binarize_map, img, roi, header, args.folder, args.file_prefix)
