from lis2.utils.file_utils import get_sorted_file_list


def test_get_sorted_file_list_returns_supported_extensions_in_order(tmp_path):
    filenames = [
        "zeta.pt",
        "gamma.fits.gz",
        "beta.fit",
        "epsilon.npy",
        "alpha.fits",
        "delta.h5",
    ]
    for filename in filenames:
        (tmp_path / filename).touch()
    (tmp_path / "ignored.csv").touch()

    assert get_sorted_file_list(tmp_path) == sorted(filenames)


def test_get_sorted_file_list_returns_empty_list_for_empty_directory(tmp_path):
    assert get_sorted_file_list(tmp_path) == []
