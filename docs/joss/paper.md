---
title: 'LIS$^2$ (Large Image Split Segmentation): A Ready-to-Use and Modular Toolbox for Large-Scale image learning.'
tags:
  - Python
  - astronomy
  - galactic filaments
  - milky way
authors:
  - name: Loris Berthelot
    orcid: 0009-0002-7993-4860
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Dominique Benielli
    orcid: 0009-0000-4293-5777
    affiliation: 1
  - name: François-Xavier Dupé
    orcid: 0000-0002-0697-2981
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1
  - name: Elliot Maitre
    affiliation: 2
  - name: Julien Rabault
    affiliation: 2
  - name: Caroline de Pourtales
    affiliation: 2
  - name: Thierry Artières
    orcid: 0000-0003-3696-0321
    affiliation: 3
  - name: Annie Zavagno
    orcid: 0000-0001-9509-7316
    affiliation: 4
affiliations:
 - name: Aix Marseille Univ, CNRS, LIS, Marseille, France
   index: 1
 - name: IRIT UMR5505 CNRS, CNRS PNRIA, Toulouse, France
   index: 2
 - name: Aix Marseille Univ, CNRS, Centrale Med, LIS, Marseille, France
   index: 3
 - name: Aix Marseille Univ, CNRS, CNES, LAM, Marseille, France
   index: 4
date: 3 February 2026
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.1051/0004-6361/202244103
# ass-doi: 10.1051/0004-6361/202450828
# aas-journal: Astromony and Astrophysics
---

# Summary

The `LIS`$^2$ toolbox is an accessible and streamlined framework designed to tackle the challenges of applying
deep learning to large-scale images, such as those produced by astronomical surveys of the Galactic plane.
Built on top of `PyTorch`, `LIS`$^2$ focuses on semantic segmentation of single, high-resolution images.
It streamlines the entire machine learning workflow—from model selection and training to inference—while
offering two key features: (1) a robust cross-validation framework for hyperparameter tuning and model comparison,
and (2) a seamless pipeline for training and inference on 2D observational data.
To manage the complexity of large-scale imaging, `LIS`$^2$ integrates a dedicated preprocessing workflow
and an optimized data storage strategy. This ensures scalability, performance, and seamless integration with
machine learning methodologies.  Although `LIS`$^2$ was initially developed for astrophysical applications, it is
flexible and easy to use. Researchers can integrate custom models and extend its capabilities to other domains.

Source code and full documentation are [available](https://github.com/multi-learn/lis2).

# Statement of need

Machine learning has become a powerful tool for tackling semantic segmentation problems [@lateef2019survey],
enabling scientists to extract meaningful information from complex data. This is typically achieved using a
dataset composed of multiple images, each paired with a corresponding segmentation mask.
However, such complete datasets are unusual. First, for example when dealing with astrophysical surveys,
we may only have one wide image of the sky. Second, masks can be time-consuming to produce, and, they may be
incomplete or contain errors [@berthelot2024supervised]. Despite these constraints, two fundamental
aspects of machine learning remain essential: (1) comparing model performance to select the best approach,
and (2) applying the trained model to real observational data.

# State of the field

A widely used approach for model comparison in machine learning is cross-validation
[@kohavi1995study; @bengio2004no], where the dataset is partitioned into training and validation
subsets to evaluate model generalization. However, standard cross-validation methods are challenging
to apply when working with a single large image. A naive splitting of a single image into smaller regions
introduces dependencies between the training and validation sets, which can potentially lead to biased evaluations.
Thus, a carefully designed data partitioning strategy is necessary to mitigate these biases while preserving the
integrity of the evaluation process. Beyond model selection, applying a trained model to real data introduces
additional complexities [@berthelot2024supervised]. Large-scale images require substantial computational
resources for both training and inference, demanding efficient preprocessing and storage solutions.

To tackle these challenges, `LIS`$^2$ provides the following: (1) a dedicated data processing strategy
optimized for machine learning on a single large image, (2) an intelligent storage system that efficiently
manages image data, and (3) a controlled and versatile environment for comparing and deploying models.
Its modular code structure enables the seamless integration of new models, training strategies, and
inference procedures. This make `LIS`$^2$ adaptable beyond semantic segmentation tasks and applicable to other
domains facing similar challenges in 2D.

# Software design

## Extracting Datasets from a Single Large Image

A critical aspect of working with large-scale images is preserving their intrinsic
spatial properties. In astrophysical images, for example, structures like filaments show significant
variations along specific axes, especially along the galactic longitude
[@hacar2023; @schisano2020hi]. In our approach [@berthelot2024supervised], the
partitioning is designed to distribute patches along this primary axis of variation rather
than randomly. To accomplish this, the image is divided into $k$ regions of equal size, ensuring
that each subset provides a distinct and independent portion of the data. This is achieved using
a sliding window approach, where small sub-images or patches, are extracted. This ensures that
each partition contains a diverse range of features, mitigating the risk of regional biases that could
affect model generalization.

To further enhance the representativeness of the extracted subsets further, patches
within each region are selected to maintain a balanced coverage of the image’s structural
diversity. This randomization process prevents artificial correlations between the training and
test sets while ensuring that all significant structures are included in the dataset. To prevent
data leakage, any overlapping samples between training and test sets are carefully removed to ensure
that no identical regions appear in both subsets. This guarantees that models are evaluated on unseen data,
providing a more reliable measure of generalization performance.

## Storing extracted Datasets

When extracting datasets and cross-validation folds, different tasks may require
different stride values. For instance, applying trained models to real observational data
is best done with a small stride, as this ensures smoother predictions by preserving continuity
across overlapping patches. In contrast, cross-validation experiments can use a larger stride to
reduce computational costs by limiting the number of extracted patches. However, generating multiple
datasets with varying strides introduces substantial redundancy because many patches overlap
between datasets, leading to unnecessary duplication and increased storage demands.

![Dataset partitioning and storage pipeline.
(A) The image is divided into patches using a stride
of 1, storing each patch with its position. (B) During
training and inference, patches are assigned to training,
validation, or test sets based on their positions,
ensuring a structured and leakage-free supervised learning process.](pipeline.png){#pipeline height="5cm"}

To address this issue, we use an optimized storage strategy in which we pre-extract all possible
patches with a stride of 1. Instead of storing multiple datasets with different stride values,
we maintain one precomputed dataset that contains all possible patches for a given patch size.
During training or evaluation, only the corresponding patches are dynamically loaded
in real-time (see Figure \ref{pipeline}). This approach yields in a non-redundant,
flexible, and memory-efficient data loader.

## Machine learning scripts

`LIS`$^2$ implements two essential machine learning procedures: (1) $k$-fold cross-validation,
which provides $k$ independent performance metrics for model comparison,
and (2) a full-segmentation pipeline. The latter trains models and infers on their
respective test folds. This ensures that each region of the image is segmented in a test setting
and prevents data leakage.

Although these components are designed for large-scale image processing, they remain adaptable
to other machine learning tasks. However, datasets must be formatted correctly, and data-loading
procedures must be adjusted accordingly.

## Versatility and re-usability of the toolbox

The `LIS`$^2$ toolbox is built on top of `PyTorch` [@paszke2019pytorch] to enable flexibility and
allow users to customize and extend functionalities as needed. The code is structured around two
main base classes that provide a configurable foundation for all components. A YAML
configuration file allows users to easily modify experimental setups without altering the core implementation.

Key pipelines and essential components are implemented as abstract classes. This allows users
to eventually introduce custom models, training procedures, or evaluation strategies (`PyTorch`
models are available as standard) without disrupting the overall code structure. Comprehensive
documentation details the roles of the base classes and the extendable components and provides
implementation guidelines to ensue smooth integration for different applications beyond astrophysics.

## Example of training configuration
The `LIS`$^2$ package provides a ready-to-use `Trainer` to orchestrate the training,
validation, and testing of models. The Trainer dynamically integrates with
the set of components configured in a YAML file.

### Config YAML

Example of  Configuration YAML
```yaml
trainer:
  output_dir: ./results
  run_name: experiment_1
  model:
    type: unet
    in_channels: 1
    out_channels: 1
  optimizer:
    type: adam
    lr: 0.001
  epochs: 50
  batch_size: 16
  metrics:
    - type: dice
```

### Outputs

The metrics and losses are saved in the directory defined by `output_dir`.
Snapshots are saved as file with the model state, the optimizers parameters and the global configuration.

### Enjoy the lightness and ease of use

Here we rebuild a trainer from a snapshot and continue the training step.
```python
from core.trainer import Trainer
trainer = Trainer.from_snapshot("./results/experiment_1/best.pt")
trainer.train()
```

### Extending the models

There exists many segmentation toolboxes dedicated to recent models or for a specific kind of data.
All these toolboxes can be integrated in ours, since adding new models just asks to build an interface
which extends the `BaseModel` class.

# Research impact statement

In this work, we introduced `LIS`$^2$, a machine learning toolbox designed to address
the challenges of semantic segmentation in large images. It offers a robust cross-validation framework,
an efficient training and inference pipeline, and a modular architecture that facilitates the seamless
integration of new models. A notable feature is its specialized pre-processing and storage strategy, ensuring
efficient data management and scalable model training.  Although `LIS`$^2$ was originally developed for
astrophysical imaging, it is  highly  versatile and applicable beyond semantic segmentation. It provides
a comprehensive solution for large-scale image-based machine learning tasks. In the future, the `LIS`$^2$
will be extended to multi-view learning [@bauvin2022summit], weakly supervised learning, and 3D learning.

# AI usage disclosure

No generative AI tools were used in the development of this software, the writing
of this manuscript, or the preparation of supporting materials.

# Acknowledgements

This work has been supported by the PNRIA program of the CNRS, the MITI program of CNRS
(Mission pour les Initiatives Transverses et Interdisciplinaires) and AMU Amidex Foundation.

# References
