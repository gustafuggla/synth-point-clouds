The code in this repository is structured as following. There are three scripts that are used to generate the data sets:

* `assemble_scenes.py`
* `sample_scenes.py`
* `synthesize_scenes.py`

The first script is used to assemble the original scenes that were used as validation data and the augmented scenes that were used for the baseline. The second script is used to create scenes using "real" ground and vegetation. The objects can be either samples from real point clouds or synthetic from mesh geometries. The third script is used to generate fully synthetic point clouds.

The paths for where the data should be saved as well as configuration for whether scenes should be augmented, mesh geometries should be used etc. is set in each script individually using constants. The files `get_sample_objects.py` and `get_mesh_objects.py` contain shared functions for loading and altering object geometries, and `place_objects.py` contain the shared placement rules. Other shared functions are located in `utils.py`. The scenes can be saved either as ASCII or in h5 format. In the h5-format, the scenes are divided into blocks in preparation for PointNet. In both cases, the point clouds consist of 3D coordinates, intensity repeated across the RGB channels, and classification.

All scripts utilize pool processing and the number of cores used is set in each script. For the sampled and synthetic scenes, the random seed is set to the scene index `i`. Explicitly setting the random seeds is necessary for the pool processing to work, since otherwise all simultaneous processes would share the same random state and result in identical scenes. Another benefit of using the scene index as a random seed is repeatability, since the scripts will generate the exact same scenes every time.