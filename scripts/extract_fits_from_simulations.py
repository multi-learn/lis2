import numpy as np
import astropy.io.fits as fits
import argparse
import os

def txt_to_fits(path):
    data = np.genfromtxt(path)
    h2 = data[:, 8]
    h2 = np.resize(h2, (1024, 1024))
    ha = data[:, 7]
    ha = np.resize(ha, (1024, 1024))
    co = data[:, 9]
    co = np.resize(co, (1024, 1024))
    c = data[:, 10]
    c = np.resize(c, (1024, 1024))
    cp = data[:, 11]
    cp = np.resize(cp, (1024, 1024))
    header = fits.PrimaryHDU().header
    header["BITPIX"] = -64
    header["NAXIS"] = 2
    header["NAXIS1"] = 1024
    header["NAXIS2"] = 1024
    header["CRPIX1"] = 0
    header["CRPIX2"] = 0
    header["CRVAL1"] = 0
    header["CRVAL2"] = 0
    header["CDELT1"] = 0.001
    header["CDELT2"] = 0.001
    header["CTYPE1"] = 'GLON-CAR'
    header["CTYPE2"] = 'GLAT-CAR'
    header["COMMENT"] = "H2"
    h2_hdu = fits.PrimaryHDU(data=h2, header=header)
    ha_hdu = fits.PrimaryHDU(data=ha, header=header)
    co_hdu = fits.PrimaryHDU(data=co, header=header)
    c_hdu = fits.PrimaryHDU(data=c, header=header)
    cp_hdu = fits.PrimaryHDU(data=cp, header=header)
    hdu_list = fits.HDUList(h2_hdu)
    header["COMMENT"] = "Ha"
    hdu_list.append(ha_hdu)
    header["COMMENT"] = "co"
    hdu_list.append(co_hdu)
    header["COMMENT"] = "c"
    hdu_list.append(c_hdu)
    header["COMMENT"] = "cp"
    hdu_list.append(cp_hdu)
    hdu_list.writeto(path[:-4] + ".fits", overwrite=True)

def get_data_file(path_to_folder):
    file_list = []
    for path, _, files in os.walk(path_to_folder):
        for file in files:
            if file.endswith("chem.txt"):
                file_list.append(os.path.join(path, file))
    return file_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract fits file from simulation report")
    parser.add_argument("path", help="path of the folder to extract reports", type=str)
    args = parser.parse_args()

    file_list = get_data_file(args.path)
    for f in file_list:
        txt_to_fits(f)