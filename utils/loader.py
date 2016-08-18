import pandas as pd
import numpy as np

class Loader():
    def __init__(self, csvfile):
        #####################
        # load data
        #####################
        self.df = pd.read_csv(csvfile, parse_dates=['Dates'])

    def process(self, use_dummies=False):
        # remove X outliers (everything thats outside 3 standard deviations)
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

            return self.df, self.features

