# kaggle_sf_crime

This code is for the Kaggle San Francisco crime challenge (https://www.kaggle.com/c/sf-crime).
It contains a data loader with preprocessing and two main files.
The first trains single classifiers and evaluates them using logloss (also used in the competition), the second one (main_search.py) uses the randomized search of sklearn for hyperparameter estimation.

To get a feel for the data, visualization.py plots some statistics about the dataset.

The first try:
- Random Forest Classifier (clf = RandomForestClassifier(max_depth=16, n_estimators=1024, n_jobs=48)) placed 580/2335 with a logloss of 2.41519 (number one entry: 1.95936)

## Data
I did not want to checkin the raw data (too big), but I also hate searching data in the future,
so I zipped the kaggle data. Just unzip data/kaggle_data.zip, and you have everything you need

## Map for visualization
map-creation: script in utils (get_map_and_save.r): [0]
use ggmap package of R, specify lat/lon box,
retreive map.

Two options:
1) save to rds file w/ gray values: use python script to reload rds and plot [1], mapdata = np.loadtxt("outputmap.txt")
2) the colored image mapfile is created by ggmap (ggmapTemp.png), can be loaded and in matplotlib set extent to lat_lon_box 

Example using the second option:
![Plotted map](https://github.com/TobiasWeis/kaggle_sf_crime/raw/master/data/map_plot.png)

[0] https://www.nceas.ucsb.edu/~frazier/RSpatialGuides/ggmap/ggmapCheatsheet.pdf

[1] https://www.kaggle.com/benhamner/sf-crime/saving-the-python-maps-file
