"""
Function for training models
"""
import numpy as np

import torch
from sklearn.metrics import f1_score
from deep_filaments.torch.models import UNet_pe, UNet_pe_atl, UNet_pe_c
from deep_filaments.metrics.metrics import dice_index, pixel_acc, positive_part_accuracy


def estimate_dice(pred, target, thresholds, missing):
    """
    Estimate Dice score for n thresholds.

    Parameters
    ----------
    pred:
        The prediction
    target:
        The targeted truth
    thresholds: list
        The different tested threshold for segmentation
    missing:
        The missing data map

    Returns
    -------
    The mean of the different DICE score.
    """
    dice = np.zeros_like(thresholds)

    for i, threshold in enumerate(thresholds):
        segmentation = (pred >= threshold).type(torch.int)
        dice[i] = dice_index(segmentation, target, missing)

    return dice


def train_loop(dataloader, model, loss_fn, optimizer, device):
    """
    Train model on the corresponding data.

    Parameters
    ----------
    dataloader:
        The loader for the training data set
    model:
        The NN model to learn
    loss_fn:
        The loss function
    optimizer:
        The optimization scheme
    device:
        The device where the computation are run

    Returns
    -------
    The current loss value and dice score.
    """
    size = len(dataloader.dataset)
    averaging_coef = 0
    loss = 0
    # thresholds from 0.2 to 0.8 (step of 0.2)
    n_thresholds = 4
    dice_thresholds = [(i + 1) * 0.2 for i in range(n_thresholds)]
    dice = np.zeros(n_thresholds)
    seg_acc = 0
    accuracy = 0
    model.train()

    for batch, sample in enumerate(dataloader):
        # compute prediction and loss
        patch = sample["patch"].to(device)
        patch_size = patch.size(0)
        if isinstance(model, UNet_pe) or isinstance(model, UNet_pe_atl) or isinstance(model, UNet_pe_c):
            positions = sample["position"]
            pe = model.position_encoding(positions).to(device)
            pred = model(patch, pe)
            del positions
            del pe
        else:
            pred = model(patch)
        del patch
        target = sample["target"].to(device)
        missing = sample["missing"].to(device)
        labelled = sample["labelled"].to(device)
        missing = missing * labelled  # Conservative way
        del labelled
        idx = (missing > 0).detach().cpu().numpy()
        temp_loss = loss_fn(missing * pred, missing * target)
        temp_acc = pixel_acc(torch.round(pred), target, missing)
        accuracy += temp_acc * idx.sum()
        loss += temp_loss.item() * idx.sum()
        dice += estimate_dice(pred, target, dice_thresholds, missing) * idx.sum()
        temp_seg_acc = positive_part_accuracy(pred, missing * target)
        seg_acc += temp_seg_acc * (missing * target).sum()
        averaging_coef += idx.sum()

        # backpropagation
        optimizer.zero_grad()
        temp_loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            temp_loss, current = loss / averaging_coef, batch * patch_size
            temp_dice = dice[1] / averaging_coef
            temp_seg_acc = seg_acc / averaging_coef
            temp_acc = accuracy / averaging_coef
            print(
                f"loss: {temp_loss:>7f}, dice: {temp_dice:>7f}, seg: {temp_seg_acc:>7f}, "
                f"acc: {temp_acc:>7f}, [{current:>5d} / {size:>5d}]"
            )

    temp_loss = loss / averaging_coef
    temp_dice = dice[1] / averaging_coef
    temp_seg_acc = seg_acc / averaging_coef
    temp_acc = accuracy / averaging_coef
    print(
        f"loss: {temp_loss:>7f}, dice: {temp_dice:>7f}, seg: {temp_seg_acc:>7f}, "
        f"acc: {temp_acc:>7f}, [{size:>5d} / {size:>5d}]"
    )

    loss /= averaging_coef
    dice /= averaging_coef

    del target
    del missing

    return loss, dice


def validation_loop(dataloader, model, loss_fn, device, name="Test"):
    """
    Test/Validate model on the provided data.

    Parameters
    ----------
    dataloader: torch.DataLoader
        The data loader with access to the patches
    model: torch.Model
        The current NN model
    loss_fn: torch.nn.Module
        The loss function
    device:
        The current device to run the computation
    name: str, optional
        The name to display

    Returns
    -------
    A tuple with the loss value and the dice score
    """
    averaging_coef = 0
    loss = 0
    acc = 0
    seg = 0
    acc_back = 0
    # thresholds from 0.1 to 0.5
    n_thresholds = 4
    dice_thresholds = [(i + 1) * 0.2 for i in range(n_thresholds)]
    dice = np.zeros(n_thresholds)

    with torch.no_grad():
        model.eval()
        for _, sample in enumerate(dataloader):
            patch = sample["patch"].to(device)
            if isinstance(model, UNet_pe) or isinstance(model, UNet_pe_atl) or isinstance(model, UNet_pe_c):
                positions = sample["position"]
                pe = model.position_encoding(positions).to(device)
                pred = model(patch, pe)
                del positions
                del pe
            else:
                pred = model(patch)
            target = sample["target"].to(device)
            missing = sample["missing"].to(device)
            back = sample["background"].to(device)
            labelled = sample["labelled"].to(device)
            missing = missing * labelled
            del labelled
            idx = (missing > 0).detach().cpu().numpy()
            if idx.any():
                loss += loss_fn(missing * pred, missing * target).item() * idx.sum()
                acc += pixel_acc(torch.round(pred), target, missing) * idx.sum()
                dice += estimate_dice(pred, target, dice_thresholds, missing) * idx.sum()
                seg += positive_part_accuracy(pred, missing * target) * (missing * target).sum()
                acc_back += positive_part_accuracy(pred, missing * back) * (missing * target).sum()
                averaging_coef += idx.sum()

    loss /= averaging_coef
    dice /= averaging_coef
    acc /= averaging_coef
    seg /= averaging_coef
    acc_back /= averaging_coef
    print(
        f"{name} Error: \n Avg loss: {loss:>8f}, Avg dice: {dice[-1]:.3f}, Avg seg: {seg:>3f}, "
        f"Avg acc: {acc:.3f}, Acc back: {acc_back:.3f}"
    )

    del target
    del missing
    del back

    return loss, dice

def One_D_train_loop(dataloader, model, optimizer, loss_fn, device):
    """
    Train model on the corresponding data.

    Parameters
    ----------
    dataloader:
        The loader for the training data set
    model:
        The NN model to learn
    loss_fn:
        The loss function
    optimizer:
        The optimization scheme
    device:
        The device where the computation are run

    Returns
    -------
    The current loss value and dice score.
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0
    for batch, samples in enumerate(dataloader):
        # Compute prediction and loss
        X = samples["data"].to(device)
        y = samples["label"].to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return train_loss / num_batches


def One_D_validation_loop(dataloader, model, loss_fn, device, name="Validation"):
    """
    Test/Validate model on the provided data.

    Parameters
    ----------
    dataloader: torch.DataLoader
        The data loader with access to the patches
    model: torch.Model
        The current NN model
    loss_fn: torch.nn.Module
        The loss function
    device:
        The current device to run the computation
    name: str, optional
        The name to display

    Returns
    -------
    A tuple with the loss value and the dice score
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    preds = torch.zeros(size)
    labels = torch.zeros(size)
    i = 0
    with torch.no_grad():
        for _, samples in enumerate(dataloader):
            X = samples["data"].to(device)
            y = samples["label"].to(device)
            batch_size = y.shape[0]
            pred = model(X)
            preds[i * batch_size : (i + 1) * batch_size] = pred
            labels[i * batch_size : (i + 1) * batch_size] = y
            test_loss += loss_fn(pred, y).item()
            i += 1

    preds = torch.round(preds)
    test_loss /= num_batches
    labels[labels >= 0.5] = 1
    labels[labels < 0.5] = 0
    acc = (preds == labels).type(torch.float).sum().item() / size
    filament_count = (preds == 1).type(torch.float).sum().item() /size
    f1 = f1_score(labels, preds)
    print(f"{name} Error: \n F1: {f1}, Accuracy: {(100*acc):>0.1f}%, Avg loss: {test_loss:>8f}, Filaments: {(100*filament_count):>0.1f}% \n")

    return test_loss, acc, f1
