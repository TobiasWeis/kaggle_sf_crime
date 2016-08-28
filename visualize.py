#!/usr/bin/python

import matplotlib.pyplot as plt
import matplotlib
from utils.loader import *
from utils.playsound_tobi import *
import seaborn as sns

l = Loader("./data/train.csv")
df, features = l.process(do_filter=True)

df["Hour"] = df.Dates.dt.hour
df["Month"] = df.Dates.dt.month
df["Year"] = df.Dates.dt.year

# overall stats
fig = plt.figure()
fig.add_subplot(121)
df["Category"].value_counts(ascending=True).plot(kind="barh", title="Number of crimes per category")
fig.add_subplot(122)
df["PdDistrict"].value_counts(ascending=True).plot(kind="barh", title="Number of crimes per PD District")

# draw plots showing the number of crimes per category for each instance of this attribute
def plotNumCrimesPerCat(attribute):
    by_param = df.groupby([attribute, 'Category'])
    table = by_param.size()
    d2table = table.unstack()
    normedtable = d2table.div(d2table.sum(1), axis=0)
    d2table.plot(kind='bar', stacked=True, figsize=(20,30), color=sns.color_palette('Set2', len(np.unique(df.Category))))
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    plt.title("Number of Crimes per Category for attribute %s" % attribute)

def plotNumCrimes(ax, attribute, normalized_per_attribute=False, normalized_per_crime=False, catlist=None):
    #plt.suptitle("Number of Crimes per Category for attribute %s" % attribute)
    by_param = None
    if catlist:
        by_param = df[df.Category.isin(catlist)].groupby([attribute, 'Category'])
    else:
        by_param = df.groupby([attribute, 'Category'])
    table = by_param.size()

    d2table = table.unstack()

    if normalized_per_attribute:
        plt.title("Normalized by number of crimes per attribute")
        normedtable = d2table.div(d2table.sum(1), axis=0)
        normedtable.plot(figsize=(20,10), color=sns.color_palette('Set2', len(np.unique(df.Category))))
    elif normalized_per_crime:
        plt.title("Normalized by total number of crimes")
        normedtable = d2table.div(d2table.sum(1).sum(0))
        normedtable.plot(figsize=(20,10), color=sns.color_palette('Set2', len(np.unique(df.Category))))
    else:
        #plt.title("Number of crimes per attribute")
        d2table.plot(ax=ax, figsize=(20,10), color=sns.color_palette('Set2', len(np.unique(df.Category))))
        
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    
import math
top_cats = ["LARCENY/THEFT", "OTHER OFFENSES", "NON-CRIMINAL", "ASSAULT", "DRUG/NARCOTIC"]
fig = plt.figure()
features = ["DayOfWeek", "PdDistrict_num", "Hour", "Year","Month","DayOfYear","StreetCorner" ]
for i,feat in enumerate(features):
    nrows = 4
    ncols = 2
    ax = fig.add_subplot(nrows, ncols, i+1)
    plotNumCrimes(ax, feat, catlist=top_cats)
#plotNumCrimes("DayOfWeek", normalized_per_attribute=True)
#plotNumCrimes("DayOfWeek", normalized_per_crime=True)
'''
plotNumCrimesPerCat("Year")
plotNumCrimesPerCat("Month")
plotNumCrimesPerCat("DayOfWeek")
plotNumCrimesPerCat("Hour")
plotNumCrimesPerCat("PdDistrict")
'''


#########################################
# Geospatial part
#########################################
# this has to be the same you chose for your map
lon_lat_box = (-122.52469, -122.33663, 37.69862, 37.82986)

# kernel density colormap used by Vasco:
# https://www.kaggle.com/vascovv/predict-west-nile-virus/west-nile-heatmap/code
alpha_cm = plt.cm.Blues
alpha_cm._init()
alpha_cm._lut[:-3,-1] = abs(np.logspace(0, 1, alpha_cm.N) / 10 - 1)[::-1]

def plot_map(cat):
    # calculate a density to plot on the map
    from sklearn.neighbors import KernelDensity
    kd = KernelDensity(bandwidth=.005)
    # coords for our chosen category
    sub = df[df["Category"] == cat]
    kd.fit(sub[['X','Y']].values)
    
    num_grid_cells = 100
    xv,yv = np.meshgrid(np.linspace(lon_lat_box[0],lon_lat_box[1],num_grid_cells), np.linspace(lon_lat_box[2],lon_lat_box[3], num_grid_cells))
    gridpoints = np.array([xv.ravel(), yv.ravel()]).T
    zv = np.exp(kd.score_samples(gridpoints)).reshape(num_grid_cells,num_grid_cells)
    
    # load the map-data created by R:ggmap (script is in utils)
    import cv2
    mapdata = cv2.cvtColor(cv2.imread("./data/outputmap.png"), cv2.COLOR_BGR2RGB)
    aspect = mapdata.shape[0] * 1.0 / mapdata.shape[1]
    plt.imshow(mapdata, 
               cmap=plt.get_cmap('gray'), 
               extent=lon_lat_box, 
               aspect=aspect,
               alpha=.8)
    
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
    plt.plot(sub.X, sub.Y, 'y.', ms=2, alpha=0.01, label=cat)
    plt.title("Category: %s" % cat)
    plt.legend()
    
    
    plt.imshow(zv, origin='lower', 
               #cmap=alpha_cm, 
               cmap="hot",
               extent=lon_lat_box, 
               aspect=aspect, alpha=0.7, interpolation='nearest')
    
    
fig = plt.figure()
new_style = {'grid': False}
matplotlib.rc('axes', **new_style)
for i,cat in enumerate(np.unique(df.Category)):
    ncols = 6 
    nrows = 7 
    fig.add_subplot(ncols, nrows, i+1)
    plot_map(cat)

plt.show()
