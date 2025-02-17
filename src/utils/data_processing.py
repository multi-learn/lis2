import os


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
        if (
            f.name.endswith(".fits")
            or f.name.endswith(".fit")
            or f.name.endswith(".fits.gz")
            or f.name.endswith(".h5")
            or f.name.endswith(".npy")
            or f.name.endswith(".pt")
        ):
            files.append(f.name)
    files.sort()
    return files
