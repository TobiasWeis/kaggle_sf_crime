#!/usr/bin/python

import matplotlib.pyplot as plt
from utils.loader import *
from utils.playsound_tobi import *
import seaborn as sns

l = Loader("./data/train.csv")
df, features = l.process(do_filter=True)

df["Hour"] = df.Dates.dt.hour
df["Month"] = df.Dates.dt.month
df["Year"] = df.Dates.dt.year

# overall stats
df["Category"].value_counts().plot(kind="bar", title="Number of crimes per category")
df["PdDistrict"].value_counts().plot(kind="bar", title="Number of crimes per PD District")

# draw plots showing the number of crimes per category for each instance of this attribute
def plotNumCrimesPerCat(attribute):
    by_param = df.groupby([attribute, 'Category'])
    table = by_param.size()
    d2table = table.unstack()
    normedtable = d2table.div(d2table.sum(1), axis=0)
    d2table.plot(kind='bar', stacked=True, figsize=(20,30), color=sns.color_palette('Set2', len(np.unique(df.Category))))
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    plt.title("Number of Crimes per Category for attribute %s" % attribute)

plotNumCrimesPerCat("Year")
plotNumCrimesPerCat("Month")
plotNumCrimesPerCat("DayOfWeek")
plotNumCrimesPerCat("Hour")
plotNumCrimesPerCat("PdDistrict")

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
    fig = plt.figure(figsize=(20,10))
    plt.imshow(mapdata, 
               cmap=plt.get_cmap('gray'), 
               extent=lon_lat_box, 
               aspect=aspect,
               alpha=.4)
    
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
    plt.plot(sub.X, sub.Y, 'y.', ms=10, alpha=0.5, label=cat)
    plt.title("Category: %s" % cat)
    plt.legend()
    
    
    plt.imshow(zv, origin='lower', 
               #cmap=alpha_cm, 
               cmap="hot",
               extent=lon_lat_box, 
               aspect=aspect, alpha=0.5, interpolation='nearest')
    
    plt.show()
    
plot_map("ARSON")
plot_map("PROSTITUTION")
plot_map("VEHICLE THEFT")
plot_map("GAMBLING")
