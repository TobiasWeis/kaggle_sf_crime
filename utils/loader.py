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
	    print "Before outlier removal: ", len(self.df)
	    self.df_outliers = self.df[np.abs(self.df.X-self.df.X.mean()) >= (3*self.df.X.std())]
	    print "Got %d outliers of coordinates, removed." % len(self.df_outliers)
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
	    # define a mapping to get some order in our plots
	    mapping = {"Monday":1, "Tuesday":2, "Wednesday":3, "Thursday":4, "Friday":5, "Saturday":6, "Sunday":7}
	    self.df = self.df.replace({'DayOfWeek' : mapping})

            #self.df['DayOfWeek_num'] = self.df['DayOfWeek'].astype('category').cat.codes.astype('float')
            self.features.append("DayOfWeek_num")

            self.df['PdDistrict_num'] = self.df['PdDistrict'].astype('category').cat.codes.astype('float')
            self.features.append("PdDistrict_num")
            
            self.df['Hour'] = self.df.Dates.dt.hour
            self.features.append("Hour")

            return self.df, self.features

