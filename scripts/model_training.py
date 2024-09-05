"""Train UNet model for filament segmentation."""
import argparse
import time

from deep_filaments.io.utils import train_procedure
from deep_filaments.utils.options import TrainParameters, DatasetParameters
import torch

if __name__ == "__main__":
    p_train_parameters = TrainParameters()
    p_data_parameters = DatasetParameters()

    parser = argparse.ArgumentParser(
        description="Learning to segment using UNet (PyTorch)."
    )
    parser.add_argument("train_path", help="The input train data (hdf5)", type=str)
    parser.add_argument("test_path", help="The input test data (hdf5)", type=str)
    parser.add_argument("--folder_name", help="Name of the folder where the models and experiments performances are going to be stored", type=str, default="data"
    )
    parser.add_argument(
        "-b",
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
        "--file_prefix",
        help="Set the file profile (default: unet_segmentation)",
        type=str,
        default="unet_segmentation",
    )
    parser.add_argument(
        "--normalization_mode",
        help="Normalize patches mode for training",
        type=str,
        default="none",
        choices=["direct", "log10", "none"],
    )
    parser.add_argument(
        "--learning_mode",
        help="Use the background/RoI only",
        choices=["oneclass", "onevsall", "conservative"],
        default="onevsall",
        type=str,
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
        choices=["UNet", "UNetPP", "SwinUNet", "DnCNN"],
    )
    args = parser.parse_args()

    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")

    t1 = time.monotonic()
    train_procedure(args.folder_name,
                    args.file_prefix,
                    args.model,
                    args.train_path,
                    args.test_path,
                    batch_size=args.batch_size,
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
    t2 = time.monotonic()

    print("Training done !")
    print("Elapsed time = {:0.2f}s".format(t2 - t1))
