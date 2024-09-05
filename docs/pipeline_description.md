This is a document describing the different steps for learning a model for filament segmentation. This presentation
includes basic commands lines.

# 1 Description of the current data sets

For the moment we have two kinds of datasets: one with mosaics and one with individual filaments. All these data are from the italian teams.

## 1.1 The mosaic maps

These data include the following (all from the italian).
1. *nh2C_mosaic[...].fits* the 37 column density files.
2. *mask_m[...]_all.fits* the 37 full overcomplete spines maps with all the detected candidates.
3. *mask_m[...]_removed.fits* the 37 filtered overcomplete spines maps.
4. *nh2C_mosaic[...]_RoISegmbranches_labels.fits* the 37 overcomplete region of interest maps (RoI).
5. *nh2C_mosaic[...]-Multi_Scale_Local_Stretch_constant.fits* the 37 normalized version of the column density maps ("constant" version).
6. *nh2C_mosaic[...]-Multi_Scale_Local_Stretch_linear.fits* the 37 normalized version of the column density maps ("linear" version).

Note: the *[...]* replace the angles.

## 1.2 Filaments dataset

This data sets of 32059 filaments includes for each filament.
1. a *subimage_m[...]_[...].fits* which is an extraction of the column density mosaic focus on the filament. The size of the image is the size of the bounding box around the region of interest.
2. a *mask_m[...]_[...].fits* which contains the region of interest of the filament and its subparts. This file contains also a header with all the extraction information (place in the NH2C maps...).

Note: the first *[...]* replace the angle of the corresponding NH2C column density file. The second *[...]* is the number id of the filament.

# 2 Data preparation

Here we describe the steps before building the training/validation/test sets that will be used for learning. For the sake of
easiness, we build several full maps (i.e. full galaxy maps).

All these maps are build using the [reproject](https://reproject.readthedocs.io/en/stable/index.html) package
from the [astropy](https://www.astropy.org/) team. This package is able to use the galactic coordinate in order
to merge several maps.

## 2.1 The column density map

As the column density maps are directly available, we can build them with the following command.
```shell
python scripts/mosaics_building.py --avoid_missing --one_file\
 nh2c_images results
```
Some explanations about the parameters.
- **--avoid_missing** option to convert all missing data (values below 0 or set to NaN) to NaN. This is important for *reproject*.
- **--one_file** option to build one huge file.
- **nh2c_images** is the directory with all the column density maps.
- **results** is the output directory which will contains the produced resulting file *merge_result.fits*.

## 2.2 The missing data map

The missing data map is a binary map (1 for the known, 0 else) which indicate which pixels must be used. We use these maps
at least to clean the column density map, avoid some noisy regions and manage the boundary. The regions to remove are defined
by bounding boxes (so we need 2 points). In order to manage many regions, they are given
by a Calc or Excel file with the following columns (one line by region to remove).
- **Tile** the angles of the mosaics (they are included in the filename, e.g. 002011).
- **Up BB l** the *l* galactic coordinate of the upper left point of the bounding box.
- **Up BB b** the *b* galactic coordinate of the upper left point of the bounding box.
- **Down BB l** the *l* galactic coordinate of the lower right point of the bounding box.
- **Down BB b** the *b* galactic coordinate of the lower right point of the bounding box.
- **Noise type** the kind of noise (0 for noise, 1 for structural noise).
- **Obs** for comments.

Then we build the missing data maps for each column density maps using the following command.
```shell
python scripts/create_missing_data_map.py nh2c_images\
 --filter list_unwanted_zones.xlsx --missing_value 0.1\
  missing-results
```
Some explanations about the parameters.
- **--filter list_unwanted_zones.xlsx** indicates the file with the regions to remove.
- **--missing_value 0.1** sets the minimal value to consider a pixel as missing (in addition to the pixel that are set to NaN).
- **nh2c_images** is the directory with all the column density maps.
- **missing-results** is the output directory which will contains the produced resulting maps.

Once these maps are produced, we can merge them to form one big map using the same script as in section [2.1](#21-the-column-density-map).
```shell
python scripts/mosaics_building.py --binarize --one_file\
 --conservative missing_images results
```
Some explanations about the new parameter.
- **--binarize** asks for a binary result (everything above 0 will be set to 1).
- **--conservative** asks for a conservative binarization, meaning only values above 0.6 are set to 1.

Note: the resulting file will still be named *merge_result.fits*. Be careful to not overwrite the previous file.

## 2.3 The filament's spines map

The full spine map is build directly using the following command.
```shell
python scripts/mosaics_building.py --binarize --one_file\
 spines_images results
```

## 2.4 Background map

The background are created from the normalized map (here the set 6 in section [1.1](#11-the-mosaic-maps)). In these maps
all pixels below a given threshold are considered to be background pixels. These thresholds are inside a Calc/Excel
file with the following columns,
- **Tile** the angles of the corresponding tile (or mosaic);
- **Threshold** the threshold for the background.

The background maps are build using the following command.
```shell
python scripts/create_background_maps.py mosaic list_background_threshold.xlsx output
```
Some explanations about the parameters.
- **mosaic** the directory with the mosaic maps.
- **list_background_threshold.xlsx** the Calc/Excel file with the threshold.
- **results** the output directory.

Now we can build the full background map with the results using the following command.
```shell
python scripts/mosaics_building.py --binarize --one_file\
 background_images results
```

# 3 Machine Learning

In this section, we focus on the machine learning process. Especially we describe the construction of the different
sets that are needed for learning.

## 3.1 Building the sets

We remind have 4 (full size) maps:
1. the column density map;
2. the missing data map (a binary map);
3. the filament's spine map (also a binary map);
4. the background map (a binary map).

The spine map forms our target, while the column density is the input. The missing data map will only be used to
avoid any computation on missing data.

We first split the maps into patches without overlapping. It is a direct split and thus for a column density patch
we have its corresponding (in term of coordinates) missing data and spine patches. These patches are saved into
an HDF5 file using the following command.
```shell
python scripts/create_patch_dataset.py density spine missing background\
 -o dataset.h5 --normalize --patch_size 64 64
```
Some explanations about the parameters.
- **density** is the name of the density map.
- **spine** is the name of the spine map.
- **missing** is the name of the missing data map.
- **background** is the name of the background map.
- **-o dataset.h5** asks to save the patches inside *dataset.h5*.
- **--normalize** asks to normalize each patch independently (min/max normalization).
- **--patch_size 64 64** asks to produce 64x64 patches (Warning: it must be the last part of the command).
- **--overlap 0** the size of the overlapping (in pixel) boundary between two patches (default: 0).

Here only the spines are used to define the labels. To integrate the background, one must add the following option,
- **--conservative** asks to use the background as 0-class truth.

Beware we use a conservative way to create the groundtruth meaning that unlabeled pixel are included in the missing data.

The patches are available in the HDF5 under the following name:
- *patches* for the density patches;
- *missing* for the missing data patches (0 for missing data);
- *targets* for the spine patches;
- *labelled* for the labelled patches (0 mean without label);
- *background* for the background patches.

Each set of the patches has the same size (in this order): number_of_patches by patch_size by patch_size by 1. The last
dimension indicate the number of channel (here 1) and is mandatory for the machine learning framework.

## 3.2 Learning

All the machine learning code relies on the [PyTorch](https://pytorch.org/) framework
which one of th main ML framework today.

The two machine learning models are
- [UNet](https://export.arxiv.org/abs/1505.04597) which is a fully convolutional network (this
means that it is independent to the size of the input).
- [UNetPP](https://github.com/MrGiovanni/UNetPlusPlus) also a fully convolutional network based on UNet.

Our task is a segmentation task (since we just want to differentiate the filaments from the background). So we ask to learn
from the patches (input) to get the targets (output). As a loss function we use the
[binary cross-entropy](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html) function which is
indicated for such task.

We remind here some vocabulary about deep learning methods with neural networks. Since the set are huge, during
the learning they are split into batch of a given size. We speak of one *epoch* when each patch has been used at least once.
So the epoch are important for convergence. The optimization algorithm is stochastic, so each run is different.
In order to control the quality of the process the input dataset is split into three sets:
- the *train* set used for parameters optimization, aka the learning step;
- the *validation* set used to control (hyperparameter selection, overtraining detection...);
- the *test* set to have some metrics about the generalization of the model.

Each set must be as independent as possible. Here the split is 80% for *train*, 10% for *validation* and 10% for *test*.

Since the size of the training set is small compared to the number of the model parameters (around 32 millions), we need
to artificially augment the size of the set. This is the purpose of the data augmentation procedure which is applied
each time the learning process asks for a patch (so the construction of the batch). To augment the data we use the following
transformation:
- *noise addition* we add a gaussian noise of zero mean and with a given variance (see options). This is a way to be less sensible to noisy zone and to regularize the model.
- *flip* we consider up-down and left-right flip.
- *rotation* we consider 90, 180 and 270 degrees rotations.

Except the noise which is always applied, the flip transformation is randomly applied (up-down and left-right are independent) and the
rotation is also randomly applied. Thus, we have an equal probability to have no rotation and no flip and to have both left-right and up-down flip with the 90 degree
rotation.

The learning process is done using the ADAM optimization scheme (with default parameters) and a fixed learning rate.
It is launched using the following script.
```shell
python scripts/unet_pytorch.py -b 64 -e 100 -d 2 --model UNet --save_best_val patch64x64_dataset.h5 results
```
Some explanations about the possible parameters.
- **-b 64** asks for batch of 64 patches.
- **-e 100** asks for 100 epochs.
- **-d 2** asks for data augmentation (0 for no augmentation, 1 for only noise injection, 2 for full augmentation).
- **--save_best_val** to save the model when the loss on the validation set has improved.
- **--no_save_best_val** if you don't want to save the model when the loss on the validation set has improved.
- **--normalize** asks for an inflight normalization.
- **--input_data_noise 0.05** the variance of the additional noise (0.05 by default) on input data.
- **--output_data_noise 0.05** the variance of the additional noise (0.05 by default) on output data.
- **--no_da_on_test** ensure that the data augmentation process is not used on the test set.
- **--model UNet** to select the neural network model (UNet or UNetPP). The default is UNet.
- **patch64x64_dataset.h5** is the set of patches inside an HDF5 file.
- **results** is the result directory where the model parameters will be saved at each run.

During the learning process we show at each epoch several metrics. These metrics are described in section [5](#5-metrics-and-validation).

In order to use the patches from Siouar, one must add the following parameter,
- **--use_newpatches** for using the other set of patches architecture.

Some parameters are specifics for the data loader and required a good understanding on how it works to use them (see the
documentation on [DataLoader](https://pytorch.org/docs/stable/data.html)),
- **--data_nb_workers 0** the number of workers used to load the data (0 by default, i.e. the main program load the data).
- **--data-prefetch 1** the number of prefetch items from the data by workers (default: 1).

Other options are also available,
- **--rng_seed 10** to change the random generator seed for the dataset splitting (default: 20).
- **--file_prefix prefix** to set the file prefix for the saved weights (default: unet_segmentation).

The learning script creates several files at the end in the *data* directory. First it creates the *experiments.hdf5* with information
of the different experiments (or runs). Second inside the directory *data/model*, it saves the learned weights of the model (in pytorch format).
The code avoids overwriting these files, so it is safe to do several runs.

# 4. Segmentation

The segmentation is direct application of the model on patches. This is done using the following scripts.
```shell
python scripts/apply_segmentation_model.py unet.pt density_map -o result --normalize\
 --missing --missmap --overlap 20 --model UNet --patch_size 32 32
```
Some explanations about the parameters.
- **unet.pt** is the file with the parameters (weights) of the model.
- **density_map** is the image to segment (in Fits format).
- **-o result** asks to save the segmentation into a file name *result*.npy (Numpy format).
- **--missing** asks to detect missing data (NaN or below a threshold).
- **--missmap** asks to deal with missing data by using a missing data map (otherwise we avoid them).
- **--normalize** asks to normalize the data before segmentation.
- **--overlap 20** asks for 20 pixels of overlapping between two patches (on order to smooth the result).
- **--patch_size 32 32** asks to use 32x32 patch (this size can be different from the patches used for learning).
- **--model UNet** to select the neural network model (UNet or UNetPP). The default is UNet.
- **--hdu_number 0** to change the index of the HDU (0 by default).

A "experimental" parallel version of the segmentation process is available with these options,
- **--new_way** asks to use the parallel machanism.
- **--batch_size 20000** asks to take a batch of 20000 patches at each steps.

Beware the parallel computation is not finish and may take longer than the classical way.

The segmentation result is im Numpy format. It can be converted (with galactic coordinate) using the following script.
```shell
python scripts/create_fits_from_npy.py result model output
```
Some explanations about the parameters.
- **result** is the input image/segmentation in Numpy format.
- **model** is the reference Fits file (we take the coordinate from this file).
- **output** is the output Fits file with the input image.

Once all the tiles are segmented, it is possible to build the full map with
```shell
python scripts/mosaics_building.py results/segmentations/full . --one_file --avoid_missing --missing_value 0.000001
```
Here all values below 0.000001 is considered as missing.

# 5. Metrics and validation

In order to validate the result, we propose different metrics. First we focus on pixel to pixel metrics using
the groundtruth. For the moment we consider the following metrics,
- **filament accuracy** the mean value of the segmentation score on the positive (filament) part. The background is not taken into account.
- **recovery accuracy** the percentage of recovered filament pixels at different thresholds (0.8, 0.5, 0.4, 0.2).
- **background accuracy** the mean value of the segmentation score on the background part.
- **background recovery accuracy** the percentage of recovered background pixels at different thresholds (0.8, 0.5, 0.4, 0.2).
Second we take a structural point of view using morphological analysis,
- **structural accuracy** the percentage of recovered filaments at different thresholds (0.8, 0.5, 0.4, 0.2).

All these metrics are computed with the following script.
```shell
python scripts/compare_segmentation.py result groundtruth background model
```
Some explanations about the parameters.
- **result** is the input image/segmentation in Numpy or Fits format.
- **groundtruth** is the reference Fits file.
- **background** is the reference background Fits file.
- **model** is the source NH2 image file.

If the HDU index number is not 0 for the groundtruth, the following option can change the index,
- **--hdu 1** for using the 1 instead of 0 as index of the HDU.
