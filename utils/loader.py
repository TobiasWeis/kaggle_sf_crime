import pandas as pd
import numpy as np

class Loader():
    def __init__(self, csvfile):
        #####################
        # load data
        #####################
        self.df = pd.read_csv(csvfile, parse_dates=['Dates'])

    def process(self, use_dummies=False, do_filter=False):
        # remove X outliers (everything thats outside 3 standard deviations)
        if do_filter:
            self.df = self.df[np.abs(self.df.X-self.df.X.mean())<=(3*self.df.X.std())]

        self.features = ["X", "Y"]

        if use_dummies:
            print "Transforming some features to dummy variables"

            # DayOfWeek names
            for f in np.unique(self.df.DayOfWeek):
                self.features.append(f)
            dummies = pd.get_dummies(self.df.DayOfWeek)
            self.df = pd.concat([self.df,dummies], axis=1) 

            # Hours
            hour = self.df.Dates.dt.hour
            for f in np.unique(hour):
                self.features.append(f)
            dummies = pd.get_dummies(hour)
            self.df = pd.concat([self.df, dummies], axis=1)

            # PdDistrict
            for f in np.unique(self.df.PdDistrict):
                self.features.append(f)
            dummies = pd.get_dummies(self.df.PdDistrict)
            self.df = pd.concat([self.df, dummies ],axis=1)

            print "Done transforming features"
        else:
            self.df['DayOfWeek_num'] = self.df['DayOfWeek'].astype('category').cat.codes.astype('float')
            self.features.append("DayOfWeek_num")

            self.df['PdDistrict_num'] = self.df['PdDistrict'].astype('category').cat.codes.astype('float')
            self.features.append("PdDistrict_num")
            
            self.df['Hour'] = self.df.Dates.dt.hour
            self.features.append("Hour")

            self.df['Month'] = self.df.Dates.dt.month
            self.features.append("Month")

            self.df['Year'] = self.df.Dates.dt.year
            self.features.append("Year")

            # discretize the coordinates
            minval = self.df.X.min()
            maxval = self.df.X.max()

            # did not improve here
            '''
            bins = 500
            labels = np.arange(0.,bins)
            self.df["X_bin"] = pd.cut(self.df.X, bins, labels=labels, include_lowest=True)
            self.features.append("X_bin")

            self.df["Y_bin"] = pd.cut(self.df.Y, bins, labels=labels, include_lowest=True)
            self.features.append("Y_bin")
            '''

            return self.df, self.features

