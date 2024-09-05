import argparse
import astropy.io.fits as fits
import numpy as np
from deep_filaments.io.dataset import OneDpixelDataset
from torch.utils.data import DataLoader, random_split
from deep_filaments.torch.models import CNN1D
import torch
from deep_filaments.torch.train import One_D_train_loop, One_D_validation_loop
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', **{'size'   : 22})
torch.manual_seed(0)

def train_on_data(
    file_prefix,
    model,
    device,
    learning_rate,
    epochs,
    dataloaders,
    directories,
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
    _, _, models_dir = directories
    train_dataloader, validation_dataloader, test_dataloader = dataloaders

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    loss_fn = torch.nn.BCELoss()
    # loss_fn = torch.nn.MSELoss()

    train_loss = np.zeros(epochs)
    val_loss = np.zeros(epochs)
    val_acc = np.zeros(epochs)
    val_f1 = np.zeros(epochs)

    ref_val_loss = 10**9  # Initialize the validation loss

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        epoch_train_loss = One_D_train_loop(train_dataloader, model, optimizer, loss_fn, device)
        epoch_val_loss, epoch_val_acc, epoch_val_f1 = One_D_validation_loop(validation_dataloader, model, loss_fn, device, name="Validation")
        if epoch_val_loss < ref_val_loss:
            ref_val_loss = epoch_val_loss
            output_model = (models_dir + "/" f"{file_prefix}_best.pt")
            torch.save(model.state_dict(), output_model)
        train_loss[t] = epoch_train_loss
        val_loss[t] = epoch_val_loss
        val_acc[t] = epoch_val_acc
        val_f1[t] = epoch_val_f1

    x = np.linspace(0, epochs, epochs)
    plt.plot(x, val_loss, label="validation loss")
    plt.plot(x, train_loss, label="training loss")
    plt.legend()
    plt.show()

    output_model = (models_dir + "/" + f"{file_prefix}_best.pt")
    model.state_dict(torch.load(output_model, map_location=device))
    model.eval()
    One_D_validation_loop(test_dataloader, model, loss_fn, device, name="Test (Best Val)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Integrate 3D cubes to get 2D data")
    parser.add_argument("input", help="The name of the 3D cube file", type=str)
    parser.add_argument("spine", help="The name of the 3D cube file", type=str)
    parser.add_argument(
        "--normalize",
        help="Normalize patches before segmentation",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--saturation",
        help="Apply tanh(ax) to the input",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--lr",
        help="Learning rate",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "-e",
        "--epochs",
        help="The number of epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--batch_size", help="The size of the batch (new way)", default=100, type=int
    )

    args = parser.parse_args()

    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    input_hdr = fits.open(args.input)
    spine_hdr = fits.open(args.spine)
    input_data = input_hdr[0].data
    spine_data = spine_hdr[0].data

    data = np.zeros((input_data.shape[1] * input_data.shape[2], input_data.shape[0])).astype(np.float32)
    for i in range(input_data.shape[0]):
        for j in range(input_data.shape[1]):
            for k in range(input_data.shape[2]):
                data[j + k * input_data.shape[1], i] = input_data[i,j,k]

    labels = np.zeros((spine_data.shape[0] * spine_data.shape[1])).astype(np.float32)
    for j in range(spine_data.shape[0]):
            for k in range(spine_data.shape[1]):
                labels[j + k * spine_data.shape[0]] = spine_data[j,k]

    train_dataset, val_dataset, test_dataset = random_split(OneDpixelDataset(data, labels, normalize=args.normalize, saturation=args.saturation), [0.7, 0.1, 0.2])
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    model = CNN1D().to(device)
    train_on_data("test", model, device, args.lr, args.epochs, (train_dataloader, test_dataloader, val_dataloader), ("","","1D_test"))