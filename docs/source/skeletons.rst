Studying skeletons
==================

.. currentmodule:: skeletons

Analyzing a cube of 3D data
------------------------------

This module provides a framework for analyzing 2D segmented masks.
The goal is to determine, using 3D data, whether a 2D mask can be decomposed into multiple submasks in 3D.

**Pipeline Overview:**

1. **Skeletonization:**
      - The first step is to skeletonize the 2D masks to reduce complexity, though the full mask can still be retained.

2. **Velocity Signal Processing:**
      - For each velocity signal from the 3D data at each skeleton point, you can choose to denoise the signal or use the raw data.

3. **Extraction of velocity:**
      - For each skeleton point, we analyze the entire 3D signal at its coordinates and extract velocities exceeding a certain threshold. Now points are in 3D.

4. **Distance Matrix Computation:**
      - Using the (optionally denoised) velocity signals and the coordinates of skeleton points, a distance matrix is computed.

5. **Clustering algorithms:**
      - The distance matrix serves as input for clustering algorithms like DBSCAN.
      - You can subdivide every cluster using `SubClustering` class assigns a unique integer label to each connected component in a binary image using `skimage.measure.label` function for example.

6. **Mask Processing & Output:**
      - The method is applied to each individual mask in the 2D segmented map.
      - The results are stored in separate FITS files.


Pipeline class
**************

Example Configuration
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

      threshold: 2.4
      speed_threshold: 15.0
      min_len_souspic: 4
      data3D_path: "data/COHRS_10p50_0p00_CUBE_3T2_R2.fit"
      data3D_reprojected_path : "data/COHRS_reprojected_10p50_0p00_CUBE_3T2_R2.fit"
      mask_toreproject_data_path: "data/spines_numbered.fits"
      clustered_data_folder: "data_test"
      skeleton_tool: 
         type : SkeletonFilFinder
      distance: 
         type : DistanceEuclieanExp
      speed_threshold: 15.0
      clustering_method: 
         type : ClusteringDBSCAN
         eps : 2.0
         min_samples : 5
      subclustering_method: 
         type : SubClusteringSkimageLabel
         min_samples : 5
         connectivity : 3
      denoising_method: 
         type : NoDenoising
      modes : ["3d_skeletons", "2d_skeletons", "all_3d_skeletons"]

To launch an extraction : 

.. code-block:: bash

   python3 scripts/main_cube_skeletonize.py -c .configs/config_clustering.yaml

ClusterCube
^^^^^^^^^^^

The pipeline is implemented in `ClusterCube` class :

.. autoclass:: src.skeletons.clustered_cube.ClusterCube
   :members:
   :undoc-members:
   :show-inheritance:


Different modes of registering clustered skeletons
**************************************************

**Available fits format are:**

1. **all_3d_skeletons:**
      - All subpart of every single skeleton is projected into 3D map and registered in ONE fits file using different integer to represent each cluster.
2. **3d_skeletons:**
      - Each cluster of each skeleton is projected into 3D map and registered into a single fits file using 0 and 1.
3. **2d_skeletons:**
      - Each cluster of each skeleton is extrated from 2D map and registered into a single fits file using 0 and 1.

Skeletonization methods
-----------------------

This module provides a collection of skeletonization algorithms.

.. currentmodule:: skeletons.skeletonize


Base class
**********

.. autoclass:: src.skeletons.skeletonize.BaseSkeletonize
   :members:
   :undoc-members:
   :show-inheritance:

Keeping the full mask
*********************

If you want to keep the full mask instead of applying a skeletonization algorithm :

.. autoclass:: src.skeletons.skeletonize.NoSkeleton
   :members:
   :undoc-members:
   :show-inheritance:

Skimage algorithm
******************

This class uses `skimage.morph.skeletonize`.

.. autoclass:: src.skeletons.skeletonize.SkeletonSkimage
   :members:
   :undoc-members:
   :show-inheritance:

FilFinder algorithm
*******************

This class uses a library called `FilFinder2D`.

To install FilFinder, run the command : ``pip install --editable git+https://github.com/e-koch/FilFinder.git#egg=Fil-Finder``

.. autoclass:: src.skeletons.skeletonize.SkeletonFilFinder
   :members:
   :undoc-members:
   :show-inheritance:

Denoising 3D signal
----------------------

Base Denoising class
********************

.. autoclass:: src.skeletons.denoising.BaseDenoising
   :members:
   :undoc-members:
   :show-inheritance:

Raw signal
**********

.. autoclass:: src.skeletons.denoising.NoDenoising
   :members:
   :undoc-members:
   :show-inheritance:

Wavelet method
**************

.. autoclass:: src.skeletons.denoising.Wavelet
   :members:
   :undoc-members:
   :show-inheritance:

Calculating distance between points of one mask/skeleton
--------------------------------------------------------

Base Distance class
*******************

.. autoclass:: src.skeletons.distance.BaseDistance
   :members:
   :undoc-members:
   :show-inheritance:

Distance between speeds
***********************

This class implements a distance function using the formula : 

.. math::

    d = \lvert v_1 - v_2 \rvert

**Parameters:**
   - **v1** (*float*): Velocity at the first point.
   - **v2** (*float*): Velocity at the second point.

.. autoclass:: src.skeletons.distance.DistanceSpeed
   :members:
   :undoc-members:
   :show-inheritance:


Spatial distance and exponential difference of speed
****************************************************

This class implements a distance function using the formula : 

.. math::

    d = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2 + (z_1 - z_2)^2} + \exp((\lvert v_1 - v_2 \rvert) - \text{speed_threshold})

**Parameters:**
   - **x1** (*float*): X-coordinate of the first point.
   - **y1** (*float*): Y-coordinate of the first point.
   - **z1** (*float*): Z-coordinate of the first point.
   - **v1** (*float*): Velocity at the first point.
   - **x2** (*float*): X-coordinate of the second point.
   - **y2** (*float*): Y-coordinate of the second point.
   - **z2** (*float*): Z-coordinate of the second point.
   - **v2** (*float*): Velocity at the second point.
   - **speed_threshold** (*float*): Threshold speed value representing the maximum difference for two points to be in the same velocity plane.

.. autoclass:: src.skeletons.distance.DistanceEuclieanExp
   :members:
   :undoc-members:
   :show-inheritance:


Clustering points inside a mask/skeleton
----------------------------------------

Base Clustering class
*********************

.. autoclass:: src.skeletons.clustering.BaseClustering
   :members:
   :undoc-members:
   :show-inheritance:

Clustering using Networkx
*************************

This clustering uses the function `find_cliques` from NetworkX to find all maximal cliques in an undirected graph. 
It generates cliques using a recursive backtracking algorithm. 
A maximal clique is a fully connected subgraph that cannot be extended by adding more nodes. 
The method returns a generator that yields each maximal clique as a list of nodes. 

.. autoclass:: src.skeletons.clustering.ClusteringSpeed
   :members:
   :undoc-members:
   :show-inheritance:


Clustering with DBSCAN
***********************

This clustering uses DBSCAN from `scikit-learn`. The DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm
is used for clustering data based on density. 
It groups closely packed points while marking sparse regions as noise. 
It requires two key parameters: eps (neighborhood radius) and min_samples (minimum points required to form a cluster). 
DBSCAN is useful for detecting clusters of arbitrary shapes and handling outliers effectively.

.. autoclass:: src.skeletons.clustering.ClusteringDBSCAN
   :members:
   :undoc-members:
   :show-inheritance:

Clustering with Agglomerative
*****************************

This clustering uses Agglomerative from `scikit-learn`. The Agglomerative clustering performs a hierarchical clustering using
a bottom up approach: each observation starts in its own cluster, and clusters are successively merged together.
The linkage criteria determines the metric used for the merge strategy.

.. autoclass:: src.skeletons.clustering.ClusteringAgglomerative
   :members:
   :undoc-members:
   :show-inheritance:

Even more clustering
********************

Base SubClustering class
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: src.skeletons.subclustering.BaseSubClustering
   :members:
   :undoc-members:
   :show-inheritance:

Connectivity inside a cluster
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This sub clustering methods allows to label indivudal component in a binary image. It uses `skimage.measure.label` function.
It assigns a unique integer label to each connected region of foreground pixels. 
The function supports different connectivity options. 
The output is a labeled array where each region has a unique identifier.

.. autoclass:: src.skeletons.subclustering.SubClusteringSkimageLabel
   :members:
   :undoc-members:
   :show-inheritance:

