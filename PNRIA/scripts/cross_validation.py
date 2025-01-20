"""
Creation of dataset from given n(h2), RoI and mask (missing data)
"""
import argparse
import os
import torch
import numpy as np

import deep_filaments.utils.informations as dfui
from deep_filaments.torch.train import train_loop, validation_loop
from deep_filaments.torch.models import UNet, UNetPP, SwinUNet, DnCNN, UNet_pe, UNet_pe_at, UNet4_pe, UNet_pe_c, \
    UNet_pe_atl

from deep_filaments.io.dataset import FilamentsDataset
from torch.utils.data import DataLoader
import deep_filaments.io.utils as utils
from deep_filaments.utils.options import TrainParameters, DatasetParameters


def final_train_procedure(
        directory,
        prefix_str,
        model_arch,
        train_dataloader,
        test_dataloader,
        learning_rate=1e-3,
        epochs=100,
        device="cpu",
        early_stopping=1e-4,
        pe="sym",
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
    elif model_arch == "UNet_pe":
        model = UNet_pe(encoding=pe)
    elif model_arch == "UNet_pe_at":
        model = UNet_pe_at()
    elif model_arch == "UNet4_pe":
        model = UNet4_pe()
    elif model_arch == "UNet_pe_c":
        model = UNet_pe_c()
    elif model_arch == "UNet_pe_atl":
        model = UNet_pe_atl()
    model.to(device)

    loss_fn = torch.nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # The scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, [10, 20, 30, 40, 50, 60, 70, 80, 90], gamma=0.1
    )

    train_loss = np.zeros(epochs)

    t = 0
    while t < epochs and (True if t < 5 else utils.early_stopping_check(train_loss[: t], early_stopping, 10)):
        print(f"Epoch {t + 1}\n-------------------------------")
        epoch_train_loss = train_loop(train_dataloader, model, loss_fn, optimizer, device)

        lr_scheduler.step()
        train_loss[t] = epoch_train_loss
        if train_loss[t] == train_loss[:t + 1].min():
            torch.save(model.state_dict(), os.path.join(directory, prefix_str + "_model.pt"))
        t += 1

    model.load_state_dict(torch.load(os.path.join(directory, prefix_str + "_model.pt"), map_location=device))

    test_loss, test_dice, test_map, test_aucroc, test_mssim = validation_loop(test_dataloader, model, loss_fn, device,
                                                                              name="Test")

    dfui.save_experiment_info(
        str(os.path.join(directory, prefix_str + "_training.hdf5")),
        train_loss,
        train_loss,
        test_loss,
        test_dice,
        test_map,
        test_aucroc,
        test_mssim,
        model_arch,
        t,
        1024,
        learning_rate
    )

    return test_dice, test_map, test_aucroc


if __name__ == "__main__":
    p_train_parameters = TrainParameters()
    p_data_parameters = DatasetParameters()
    parser = argparse.ArgumentParser(description="Patch dataset construction.")
    parser.add_argument("train_dataset_path", help="The path to the hdf5 train dataset ", type=str)
    parser.add_argument("validation_dataset_path", help="The path to the hdf5 validation dataset ", type=str)
    parser.add_argument("test_dataset_path", help="The path to the hdf5 test dataset ", type=str)
    parser.add_argument("--folder", help="The output database file", type=str, default="k_fold"
                        )
    parser.add_argument(
        "--batch_size",
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
        "--file_prefix",
        help="Set the file profile (default: ML_segmentation)",
        type=str,
        default="ML_segmentation",
    )
    parser.add_argument(
        "--model",
        help="The network model",
        default="UNet",
        type=str,
        choices=["UNet", "UNetPP", "SwinUNet", "DnCNN", "UNet_pe", "UNet4_pe", "UNet_pe_at", "UNet_pe_c", "UNet_pe_atl",
                 "UNet_pe_full"],
    )
    parser.add_argument(
        "--early_stopping",
        help="Loss delta before early stopping",
        default=1e-4,
        type=float
    )
    parser.add_argument(
        "--pe",
        help="Postion encoding strategy for UNet_pe",
        default="sym",
        type=str,
        choices=["sym", "lin", "sin"],
    )

    args = parser.parse_args()

    os.system(f"mkdir {args.folder}")
    os.system(f"mkdir {os.path.join(args.folder, 'models')}")
    os.system(f"mkdir {os.path.join(args.folder, 'models', 'cross_val')}")
    os.system(f"mkdir {os.path.join(args.folder, 'models', 'final_models')}")

    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")

    train_dataset_files = utils.get_sorted_file_list(args.train_dataset_path)
    validation_dataset_files = utils.get_sorted_file_list(args.validation_dataset_path)
    test_dataset_files = utils.get_sorted_file_list(args.test_dataset_path)

    learning_rate = [5e-3, 1e-3, 5e-4, 1e-4, 5e-5]

    for i in range(len(train_dataset_files)):
        print("*" * 20)
        print(f"Fold {i} - hyperparameters choice")
        print("*" * 20)
        geometric_perf = torch.zeros((len(learning_rate)))
        train_dataset = FilamentsDataset(dataset_path=os.path.join(args.train_dataset_path, train_dataset_files[i]),
                                         data_augmentation=args.data_augmentation,
                                         input_data_noise=args.input_data_noise,
                                         output_data_noise=args.output_data_noise)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        validation_dataset = FilamentsDataset(
            dataset_path=os.path.join(args.validation_dataset_path, validation_dataset_files[i]), data_augmentation=0,
            input_data_noise=args.input_data_noise, output_data_noise=args.output_data_noise)
        validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True)
        for j in range(len(learning_rate)):
            print("*" * 20)
            print(f"Learning rate: {learning_rate[j]}")
            print("*" * 20)
            test_dice, test_map, test_aucroc = utils.train_procedure(os.path.join(args.folder, "models", "cross_val"),
                                                                     args.file_prefix + f"_fold_{i}_cross_val_lr_{learning_rate[j]}",
                                                                     args.model,
                                                                     train_dataloader,
                                                                     validation_dataloader,
                                                                     learning_rate=learning_rate[j],
                                                                     epochs=args.epochs,
                                                                     device=device,
                                                                     early_stopping=args.early_stopping,
                                                                     pe=args.pe,
                                                                     )
            geometric_perf[j] = (test_dice * test_map * test_aucroc) ** (1 / 3)
        print("*" * 20)
        print(f"Fold {i} - Final training")
        print("*" * 20)
        test_dataset = FilamentsDataset(dataset_path=os.path.join(args.test_dataset_path, test_dataset_files[i]),
                                        data_augmentation=0, input_data_noise=args.input_data_noise,
                                        output_data_noise=args.output_data_noise)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
        validation_dataset.data_augmentation = args.data_augmentation
        train_dataloader = DataLoader(torch.utils.data.ConcatDataset([train_dataset, validation_dataset]),
                                      batch_size=args.batch_size, shuffle=True)
        best_idx = torch.argmax(geometric_perf)
        final_train_procedure(os.path.join(args.folder, "models", "final_models"),
                              args.file_prefix + f"_fold_{i}_final_lr_{learning_rate[best_idx]}",
                              args.model,
                              train_dataloader,
                              test_dataloader,
                              learning_rate=learning_rate[best_idx],
                              epochs=args.epochs,
                              device=device,
                              early_stopping=args.early_stopping,
                              pe=args.pe,
                              )