# Deepfilaments

A python ML project which use deep learning to detect and better understand filaments in the galaxy.

## Requirements

1. Python (>=3.8)
2. numpy
3. scipy
4. matplotlib
5. h5py
6. astropy
7. reproject
8. scikit-image
9. scikit-learn
11. pandas
12. torch
13. seaborn
14. openpyxl
15. numba
16. networkx
17. tqdm
18. einops
19. timm

To install these package you can use Conda (with development tools)
```
conda env create -f environment.yml
```
or if you prefer *pipenv* (with *--dev* for the development tools)
```
pipenv install
```

*Note*: this code has only be tested on python 3.8 and 3.9.

## Development Guidelines
Use pre-commit hooks to check whether your changes comply with the recommended code style guidelines.
After installation of the required packages via conda, install the git hook scripts using
```
pre-commit install
```
This package will automatically check your changes upon every attempt to commit them.

To install `deepfilaments` in the develop mode run
```
pip install -e .
```

## Get started
To segment your own column density maps:
### Create a dataset
```shell
python scripts/create_dataset.py path/to/column_density_map.fits -o path/to/output/output_name --patch_size 32 32 --test_overlap 31
```
Some explanations about the parameters:
- **path/to/column_density_map.fits** path to the .fits file to segment.
- **-o path/to/output/output_name** path to the dataset output file which will be use for segmentation.
- **--patch_size 32 32** The original map will be split into small patches of size 32x32. Using a different value is possible but we expect 32x32 to perform the best since our model was trained with this size.
- **--test_overlap 31** overlap between small patches. To have the smoothest segmentation, using *patch_size - 1* is recommanded. The value can be lowered to reduce the computation time.

### Apply trained models
```shell
python scripts/segmentation.py path/to/model.h5 path/to/create_dataset_output.h5 path/to/column_density_map.fits -o path/to/output/output_name.fits --normalization_mode direct --model UNet --batch_size 1000
```
Some explanations about the parameters:
- **path/to/model.h5** path to the neural network file to use for segmentation.
- **path/to/create_dataset_output.h5** path to the dataset.h5 file created from the previous script.
- **path/to/column_density_map.fits** path to the .fits file to segment.
- **-o path/to/output/output_name.fits** path to the .fits output file (the segmentation).
- **--normalization_mode direct** time of normalization to used. From our experiements, this does not have a significant influence on results. Can be *direct* or *log10*.
- **--model UNet** Neural network architecture to use.
- **--batch_size 1000** Batch size. Does not have any influence on the results. The bigger, the faster.

## Authors

- François-Xavier Dupé (LIS)
- Annie Zavagno (LAM)
- Siouar Bensaid (LAM/LIS)
- Loris Berthelot (LAM/LIS)

## Links
- [LAM](www.lam.fr)
- [LIS](www.lis-lab.fr)
