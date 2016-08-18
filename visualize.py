#!/usr/bin/python

import matplotlib.pyplot as plt
from utils.loader import *
from utils.playsound_tobi import *

l = Loader("./data/train.csv")
df, features = l.process()

# this has to be the same you chose for your map
lon_lat_box = (-122.52469, -122.33663, 37.69862, 37.82986)

# calculate a density to plot on the map
from sklearn.neighbors import KernelDensity
kd = KernelDensity(bandwidth=2.)
kd.fit(df[['X','Y']].values)

num_grid_cells = 10
xv,yv = np.meshgrid(np.linspace(lon_lat_box[0],lon_lat_box[1],num_grid_cells), np.linspace(lon_lat_box[2],lon_lat_box[3], num_grid_cells))
gridpoints = np.array([xv.ravel(), yv.ravel()]).T
zv = kd.score_samples(gridpoints).reshape(num_grid_cells,num_grid_cells)


# load the map-data created by R:ggmap (script is in utils)
import cv2
mapdata = cv2.cvtColor(cv2.imread("./data/outputmap.png"), cv2.COLOR_BGR2RGB)
aspect = mapdata.shape[0] * 1.0 / mapdata.shape[1]
plt.imshow(mapdata, 
           cmap=plt.get_cmap('gray'), 
           extent=lon_lat_box, 
           aspect=aspect)

# Those are all categories present in the dataset
'''
'ARSON', 'ASSAULT', 'BAD CHECKS', 'BRIBERY', 'BURGLARY',
       'DISORDERLY CONDUCT', 'DRIVING UNDER THE INFLUENCE',
       'DRUG/NARCOTIC', 'DRUNKENNESS', 'EMBEZZLEMENT', 'EXTORTION',
       'FAMILY OFFENSES', 'FORGERY/COUNTERFEITING', 'FRAUD', 'GAMBLING',
       'KIDNAPPING', 'LARCENY/THEFT', 'LIQUOR LAWS', 'LOITERING',
       'MISSING PERSON', 'NON-CRIMINAL', 'OTHER OFFENSES',
       'PORNOGRAPHY/OBSCENE MAT', 'PROSTITUTION', 'RECOVERED VEHICLE',
       'ROBBERY', 'RUNAWAY', 'SECONDARY CODES', 'SEX OFFENSES FORCIBLE',
       'SEX OFFENSES NON FORCIBLE', 'STOLEN PROPERTY', 'SUICIDE',
       'SUSPICIOUS OCC', 'TREA', 'TRESPASS', 'VANDALISM', 'VEHICLE THEFT',
       'WARRANTS', 'WEAPON LAWS'
'''

# Choose a specific category to plot incidents
cat = 'DRUNKENNESS'
cat2 = 'VEHICLE THEFT'
plt.plot(df[df.Category == cat2].X, df[df.Category == cat2].Y, 'r.', ms=10, alpha=0.5, label=cat2)
plt.plot(df[df.Category == cat].X, df[df.Category == cat].Y, 'b.', ms=10, alpha=0.5, label=cat)
plt.legend()

'''
plt.imshow(zv, origin='lower', 
           #cmap=alpha_cm, 
           cmap="cool",
           extent=lon_lat_box, 
           aspect=aspect, alpha=0.5)
'''
plt.show()

