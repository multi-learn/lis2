"""
Transformation functions for data augmentation
"""
import numpy as np


def transform(data, num_tf):
    """
    Transform the data using a given transformation

    Parameters
    ----------
    data: numpy.ndarray
        The data to transform
    num_tf: int
        The number of the transformation

    Returns
    -------
    The transformed data
    """
    shape = data.shape
    data = np.squeeze(data)

    if num_tf == 1:
        data = np.fliplr(data)
    elif num_tf == 2:
        data = np.flipud(data)
    elif num_tf == 3:
        data = np.rot90(data)
    elif num_tf == 4:
        data = np.rot90(data, 2)
    elif num_tf == 5:
        data = np.rot90(data, 3)
    elif num_tf == 6:
        data = np.fliplr(np.flipud(data))
    elif num_tf == 7:
        data = np.fliplr(np.rot90(data))
    elif num_tf == 8:
        data = np.fliplr(np.rot90(data, 2))
    elif num_tf == 9:
        data = np.fliplr(np.rot90(data, 3))
    elif num_tf == 10:
        data = np.flipud(np.rot90(data))
    elif num_tf == 11:
        data = np.flipud(np.rot90(data, 2))
    elif num_tf == 12:
        data = np.flipud(np.rot90(data, 3))
    elif num_tf == 13:
        data = np.fliplr(np.flipud(np.rot90(data)))
    elif num_tf == 14:
        data = np.fliplr(np.flipud(np.rot90(data, 2)))
    elif num_tf == 15:
        data = np.fliplr(np.flipud(np.rot90(data, 3)))

    return np.reshape(data.copy(), shape)


def apply_noise_transform(data_list, input_noise_var=0.05, output_noise_var=0.05):
    """
    Apply a random transform to a list of data (full_version)
    Note: only apply on the first item

    Parameters
    ----------
    data_list: list
        A list of data
    input_noise_var: float
        The variance of the additional noise (input)
    output_noise_var: float
        The variance of the additional noise (output)

    Returns
    -------
    A list of transform data
    """
    in_noise = np.array(
        np.random.standard_normal(data_list[0].shape) * input_noise_var, dtype="f"
    )
    out_noise = np.array(
        np.random.standard_normal(data_list[0].shape) * output_noise_var, dtype="f"
    )
    n_in_data = data_list[0] + in_noise
    n_out_data = data_list[1] + out_noise
    res = data_list.copy()
    res[0] = n_in_data
    res[1] = n_out_data
    return res


def apply_extended_transform(data_list, rng, noise_var):
    """
    Apply a random transform to a list of data with noise everywhere (extended version)
    Parameters
    ----------
    data_list: list
        A list of data
    rng: random.Random
        A random generator
    noise_var: list[float]
        The variance of the additional noise for each element

    Returns
    -------
    A list of transform data
    """
    n_tf = rng.randint(0, 15)
    res = []
    for data, variance in zip(data_list, noise_var):
        noise = np.array(np.random.standard_normal(data.shape) * variance, dtype="f")
        data = data + noise
        res.append(transform(data, n_tf))
    return res
