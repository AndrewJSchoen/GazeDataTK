import math, os, re, sys, json, logging, datetime
from math import factorial
import numpy as np
import pandas
from matplotlib import pyplot as plt
from matplotlib import patches as ptc
from PIL import Image, ImageDraw
from matplotlib import animation
import scipy.signal as spsig
from scipy import misc
from numpy import arange, cos, linspace, pi, sin, random, linalg
from collections import OrderedDict

def exists(path):
    if os.path.exists(cleanPathString(path)):
        return 1
    else:
        return 0

def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def byteify(input):
    if isinstance(input, dict):
        return {byteify(key):byteify(value) for key, value in input.iteritems()}
    elif isinstance(input, list):
        return [ byteify(element) for element in input ]
    elif isinstance(input, unicode):
        return input.encode('utf-8')
    else:
        return input

def stringify(n, unfloatify = False):
    if unfloatify == True:
        try:
            n = int(n)
        except:
            pass

    x = str(n)
    return x

def floatify(s):
    try:
        x = float(s)
    except:
        x = s

    return x

def cleanPathString(path):
    if path.endswith('\n'):
        path = path[:-1]
    if path.endswith('/'):
        path = path[:-1]
    if path.startswith('='):
        path = path[1:]
    if path.startswith('~'):
        path = os.path.expanduser(path)
    realpath = os.path.realpath(path)
    return realpath

def read_gaze(csvfile, **kwargs):
    gaze = Gaze.from_file(csvfile, **kwargs)
    return gaze

def read_timing(csvfile, **kwargs):
    timing = Timing.from_file(csvfile, **kwargs)
    return timing

def remove_duplicates(inputlist):
    result = []
    for i in inputlist:
        if not i in result:
            result.append(i)
    return result

class Timing(object):

    def __init__(self, timingframe):
        #print(timingframe)
        columns = list(timingframe.columns)
        othercolumns = [x for x in columns if x not in ["STIMULUS", "TRIALINDEX"]]

        self.MetaData = [x for x in othercolumns if len(x.split("."))<2]
        self.Epochs = list(set(zip(*[x.split(".") for x in othercolumns if x not in self.MetaData])[0]))

        self.DataFrame = timingframe

        for epoch in self.Epochs:
            for endpoint in ["ONSET", "OFFSET"]:
                targetcolumn = epoch + '.' + endpoint
                if targetcolumn in columns:
                    self.DataFrame[targetcolumn] = self.DataFrame[targetcolumn]
                else:
                    self.DataFrame[targetcolumn] = None

        self.DataFrame["TRIAL.ONSET"] = self.DataFrame.apply(lambda row: row[[epoch+".ONSET" for epoch in self.Epochs]].min(),axis=1)


    @classmethod
    def from_dataframe(cls, df, stimulus=None, subject=None, trialindex=None):
        df = df.reset_index(drop=True)
        # print("DF:\n",df)
        #Check for a starting column
        if stimulus == None and trialindex == None:
            raise IOError("You must provide and designate a stimulus and/or a trialindex column.")

        columns = list(df.columns)

        #Initialize the intermediate
        initialcolumns = {}
        if stimulus != None:
            initialcolumns["STIMULUS"]=df[stimulus]
        if subject != None and subject in columns:
            initialcolumns["SUBJECT"]=df[subject]
        elif subject != None:
            initialcolumns["SUBJECT"]=SUBJECT
        if trialindex != None:
            initialcolumns["TRIALINDEX"]=trialindex

        intermediate = pandas.DataFrame(initialcolumns)
        intermediatecolumns = list(intermediate.columns)
        if "TRIALINDEX" not in intermediatecolumns and stimulus in intermediatecolumns:
            slices = []
            groups = [intermediatecolumn for intermediatecolumn in intermediatecolumns if intermediatecolumn != "STIMULUS"]
            for index, dataslice in intermediate.groupby(groups):
                frame = dataslice.drop_duplicates().reset_index(drop=True)
                frame["TRIALINDEX"] = stimulusframe.index+1
                slices.append(frame)
            stimulusframe = pandas.concat(slices)
            intermediate = pandas.merge(intermediate, stimulusframe)


        # if stimulus != None and trialindex != None and stimulus in columns and trialindex in columns:
        #     intermediate = pandas.DataFrame({"STIMULUS":df[stimulus], "TRIALINDEX":df[trialindex]}, index=df.index)
        # elif stimulus != None and stimulus in columns:
        #     intermediate = pandas.DataFrame({"STIMULUS":df[stimulus]}, index=df.index)
        #     #Add a TRIALINDEX Column
        #     stimulusframe = intermediate[["STIMULUS"]].drop_duplicates().reset_index(drop=True)
        #     stimulusframe["TRIALINDEX"] = stimulusframe.index+1
        #     intermediate = pandas.merge(intermediate, stimulusframe)
        # elif trialindex != None and trialindex in columns:
        #     intermediate = pandas.DataFrame({"TRIALINDEX":df[trialindex]}, index=df.index)
            #Cannot determine stimulus ad-hoc

        #Get non-stimulus/non-index columns, and pass them in to the intermediate frame.
        othercolumns = [x for x in columns if x not in [stimulus, trialindex]]
        for othercolumn in othercolumns:
            intermediate[othercolumn.upper()] = df[othercolumn]

        try:
            timing = Timing(intermediate)
        except:
            raise IOError("Not able to initialize Timing object!")
            timing = None

        return timing

    @classmethod
    def from_file(cls, csvfile, **kwargs):
        try:
            df = pandas.read_table(cleanPathString(csvfile), sep=",")
        except:
            raise IOError("File '{0}' could not be found!".format(cleanPathString(csvfile)))

        timing = cls.from_dataframe(df, **kwargs)

        return timing

    def to_csv(self, outfile):
        self.DataFrame.to_csv(outfile, index=False)

class GazeSubset(object):
    def __init__(self, gaze, **kwargs):
        self.parent = gaze
        #query = []
        # for key in kwargs.keys():
        #     if key in list(gaze.DataFrame.columns):
        #         query.append('{0} == {1}'.format(key, kwargs[key]))
        # querystring = " and ".join(query)
        # print(querystring)
        tempframe = gaze.DataFrame
        for kwarg in kwargs.keys():
            if kwarg.upper() in list(gaze.DataFrame.columns):
                #print(kwarg.upper(), kwargs[kwarg])
                tempframe = tempframe.loc[tempframe[kwarg.upper()] == kwargs[kwarg]]
                #print(tempframe)
        #self.DataFrame = gaze.DataFrame.query(querystring)
        self.DataFrame = tempframe
        self.Frames = len(self.DataFrame)

    def animate_init(self, dot, bar, aoicollection):
        self.Dot = dot
        #print(self.DataFrame)
        self.TimeBar = bar
        self.AOICollection = aoicollection
        if self.LocationType == "Pixels":
            self.DisplayCoordinates = zip(list(self.DataFrame['XSERIES']), list(self.DataFrame['YSERIES']))
        else:
            self.DisplayCoordinates = zip(list(self.DataFrame['XSERIES']*self.Screen.Width), list(self.DataFrame['YSERIES']*self.Screen.Height))
        self.Dot.center = (-100, -100)
        self.TimeBar.set_width(0)
        self.Duration = self.Frames / self.Hertz

    def animate_set(self):
        self.Dot.center = (-100, -100)
        self.TimeBar.set_width(0)
        return (self.Dot, self.TimeBar)

    def animate_gaze(self, i):
        self.TimeBar.set_width(self.Screen.Width * (float(i) / float(self.Frames)))
        if self.DisplayCoordinates[i][0] != None and self.DisplayCoordinates[i][1] != None:
            self.Dot.center = self.DisplayCoordinates[i]

            if self.AOICollection.is_inside(x=self.DisplayCoordinates[i][0]/self.Screen.Width, y=self.DisplayCoordinates[i][1]/self.Screen.Height):
                self.Dot.set_facecolor('green')
            else:
                self.Dot.set_facecolor('red')
        else:
            self.Dot.center = (-100, -100)
        return (self.Dot, self.TimeBar)

    def __getattr__(self, name):
        try:
            return getattr(self, name)
        except:
            try:
                return getattr(self.parent, name)
            except AttributeError, e:
                raise AttributeError("GazeSubset' object has no attribute '{0}'".format(name))

class Gaze(object):
    #Hidden Methods
    def __init__(self, gazeframe, screen=None, accepted_quality=(0,100), transfercolumns=[]):
        self.DataFrame = gazeframe
        self.VisibleColumns = ["SUBJECT", "STIMULUS", "TRIALINDEX", "TIMESTAMP", "TRIALTIMESTAMP", "XSERIES", "YSERIES", "PUPILDIAMETER", "VALIDITY", "ONSCREEN"]
        self.UtilityColumns = ["XSERIES_R", "XSERIES_L",
                               "YSERIES_R", "YSERIES_L",
                               "VALIDITY_R", "VALIDITY_L",
                               "ONSCREEN_R","ONSCREEN_L",
                               "PUPILDIAMETER_R", "PUPILDIAMETER_L",
                               "XSERIES_ORIG", "XSERIES_R_ORIG", "XSERIES_L_ORIG",
                               "YSERIES_ORIG", "YSERIES_R_ORIG", "YSERIES_L_ORIG",
                               "PUPILDIAMETER_ORIG", "PUPILDIAMETER_R_ORIG", "PUPILDIAMETER_L_ORIG",
                               "VALIDITY_ORIG", "VALIDITY_R_ORIG", "VALIDITY_L_ORIG",
                               "ONSCREEN_ORIG", "ONSCREEN_R_ORIG", "ONSCREEN_L_ORIG"]
        for column in self.VisibleColumns + self.UtilityColumns:
            if column not in list(self.DataFrame.columns.values):
                self.DataFrame[column] = None
        self.LR_Measures = []
        self.Measures = []
        self.__detect_measures__()
        self.AcceptedQuality = accepted_quality
        self.Interpolated = False
        for measure in self.Measures:
            #Update the measures
            self.__recalculate__(measure, reset=True)
        self.__detect_location_type__()
        self.__detect_hertz__()
        if screen != None:
            self.set_screen(screen)

        self.Frames = len(self.DataFrame)
        self.MetaData = transfercolumns
        self.AOIs = ["ANYAOI"]
        self.DataFrame["ANYAOI"] = False
        self.Epochs = []

        #print(self.DataFrame)

    def __repr__(self):
        return self.DataFrame

    def __str__(self):
        return self.DataFrame

    def __detect_hertz__(self):
        timestampframe = pandas.DataFrame({"TIMESTAMP":self.DataFrame["TIMESTAMP"],"NEXT_TIMESTAMP":self.DataFrame["TIMESTAMP"].shift(-1)})
        timestampframe["ELAPSED_TIME"] = timestampframe["NEXT_TIMESTAMP"]-timestampframe["TIMESTAMP"]
        average_elapsed_time = timestampframe["ELAPSED_TIME"].median()/100
        self.Hertz = 1/average_elapsed_time

    def __detect_location_type__(self):
        if 0 <= self.DataFrame["XSERIES"].mean() <= 1 and 0 <= self.DataFrame["YSERIES"].mean() <= 1:
            print("Auto-detecting that the Location data you specified is a percentage of the screen real estate.\nIf this is incorrect, run GazeObject.location_is_pixels().")
            self.LocationType = "Percent"

        else:
            print("Auto-detecting that the Location data you specified are pixels on the screen real estate.\nIf this is incorrect, run GazeObject.location_is_percent().")
            self.LocationType = "Pixels"

    def __detect_measures__(self):
        for measure in ["VALIDITY", "ONSCREEN", "XSERIES", "YSERIES", "PUPILDIAMETER"]:
            if np.all(pandas.notnull(self.DataFrame[measure+"_R_ORIG"])) and np.all(pandas.notnull(self.DataFrame[measure+"_L_ORIG"])):
                self.LR_Measures = remove_duplicates(self.LR_Measures + [measure])
                self.Measures = remove_duplicates(self.Measures + [measure])
            elif np.all(pandas.notnull(self.DataFrame[measure+"_ORIG"])):
                self.Measures = remove_duplicates(self.Measures + [measure])

    def __recalculate__(self, measure, reset=False):
        print("  Recalculating {0}".format(measure))
        columnvals = list(self.DataFrame.columns.values)
        if measure in self.LR_Measures:
            #If the L/R measure columns don't exist, create them from the origs.
            if reset==True:
                self.DataFrame[measure+"_R"] = self.DataFrame[measure+"_R_ORIG"]
                self.DataFrame[measure+"_L"] = self.DataFrame[measure+"_L_ORIG"]

            if measure != "VALIDITY" and measure != "ONSCREEN":

                #Calculate the revised L/R measures, then regenerate the average.
                if "VALIDITY" in self.LR_Measures:
                    self.DataFrame[measure+"_R"] = np.apply_along_axis(lambda row: row[0] if (self.AcceptedQuality[0] <= row[1] <= self.AcceptedQuality[1] and row[2] == True) or self.Interpolated == True else np.nan, axis=1, arr=self.DataFrame.as_matrix(columns=[measure+"_R", "VALIDITY_R", "ONSCREEN_R"]))
                    self.DataFrame[measure+"_L"] = np.apply_along_axis(lambda row: row[0] if (self.AcceptedQuality[0] <= row[1] <= self.AcceptedQuality[1] and row[2] == True) or self.Interpolated == True else np.nan, axis=1, arr=self.DataFrame.as_matrix(columns=[measure+"_L", "VALIDITY_L", "ONSCREEN_L"]))
                elif "VALIDITY" in self.Measures:
                    self.DataFrame[measure+"_R"] = np.apply_along_axis(lambda row: row[0] if (self.AcceptedQuality[0] <= row[1] <= self.AcceptedQuality[1] and row[2] == True) or self.Interpolated == True else np.nan, axis=1, arr=self.DataFrame.as_matrix(columns=[measure+"_R", "VALIDITY", "ONSCREEN_R"]))
                    self.DataFrame[measure+"_L"] = np.apply_along_axis(lambda row: row[0] if (self.AcceptedQuality[0] <= row[1] <= self.AcceptedQuality[1] and row[2] == True) or self.Interpolated == True else np.nan, axis=1, arr=self.DataFrame.as_matrix(columns=[measure+"_L", "VALIDITY", "ONSCREEN_L"]))

            if measure == "ONSCREEN":
                #If both eyes are onscreen, set the "average" onscreen to be true as well. Otherwise, false.
                self.DataFrame[measure] = np.apply_along_axis(lambda row: True if row[0] == True or row[1] == True else False, 1, self.DataFrame.as_matrix(columns=[measure+"_R", measure+"_L"]))
            else:
                #Perform a simple mean, ignoring NaNs
                self.DataFrame[measure] = self.DataFrame[[measure+"_R", measure+"_L"]].mean(axis=1)

        elif measure in self.Measures:
            if reset==True:
                #If the measure column doesn't exist, create it from the origs.
                self.DataFrame[measure] = self.DataFrame[measure+"_ORIG"]
            if "VALIDITY" in self.Measures and measure != "VALIDITY" and measure != "ONSCREEN":
                self.DataFrame[measure] =  self.DataFrame[[measure, "VALIDITY", "ONSCREEN"]].apply(lambda row: row[measure] if (self.AcceptedQuality[0] <= row["VALIDITY"] <= self.AcceptedQuality[1] and row["ONSCREEN"] == True) or self.Interpolated == True else None, axis=1)
                #self.DataFrame[measure] = self.DataFrame[[measure]].mask(self.DataFrame.apply(lambda row: False if (self.AcceptedQuality[0] <= row["VALIDITY"] <= self.AcceptedQuality[1] and row['ONSCREEN'] == True) or self.Interpolated == True else True, axis=1))

        else:
            raise IOError("That measure is not included in this gaze data.")

    def __row_epoch_inside__(self, timestamp, next_timestamp, trialindex, next_trialindex, epoch_onset = None, epoch_offset = None):
        inside = False

        if epoch_onset == None:
            epoch_onset = 0
        if epoch_offset == None:
            epoch_offset = 10000000000
        #print("\nTimestamp: {0}\nNext_Timestamp: {1}\nTrialIndex: {2}\nNext_TrialIndex: {3}\nEpoch_Onset: {4}\nEpoch_Offset: {5}".format(timestamp, next_timestamp, trialindex, next_trialindex, epoch_onset, epoch_offset))
        if epoch_onset <= timestamp <= epoch_offset:
            inside = True
        elif epoch_onset <= next_timestamp <= epoch_offset and int(trialindex) == int(next_trialindex) and math.fabs(timestamp - epoch_onset) < math.fabs(next_timestamp - epoch_onset):
            inside = True
        # if inside == True:
        #     print("VALID!")
        return inside

    # def __calculate_anyaoi__(self, row):
    #     if row["ANYAOI"] == True or True in row[self.AOIs].unique():
    #         return True
    #     else:
    #         return False
    #
    # def __update_anyaoi__(self):
    #     self.DataFrame["ANYAOI"] = self.DataFrame[self.AOIs].apply(self.__calculate_anyaoi__, axis=1)

    #Non-Hidden Methods
    @classmethod
    def from_dataframe(cls, df, accepted_quality=(0,100), screen=None, subject=None, stimulus=None, right_x=None, right_y=None, left_x=None, left_y=None, average_x=None, average_y=None, right_validity=None, left_validity=None, average_validity=None, right_pupildiameter=None, left_pupildiameter=None, average_pupildiameter=None, transfercolumns=[], timestamp=None):
        dfcolumns = list(df.columns.values)
        df = df.reset_index(drop=True)
        intermediate = pandas.DataFrame({"STIMULUS":df[stimulus], "TIMESTAMP":df[timestamp]}, index=df.index)

        #SUBJECT Column
        if subject in dfcolumns:
            intermediate["SUBJECT"] = df[subject]
        else:
            intermediate["SUBJECT"] = subject

        #Add a TRIALINDEX Column
        components = []
        for index, subframe in intermediate.groupby("SUBJECT"):
            stimulusframe = subframe[["SUBJECT","STIMULUS"]].drop_duplicates().reset_index(drop=True)
            stimulusframe["TRIALINDEX"] = stimulusframe.index+1
            components.append(stimulusframe)
        intermediate = pandas.merge(intermediate, pandas.concat(components))

        #Add a TRIALTIMESTAMP Column
        components = []
        for index, subframe in intermediate.groupby(["SUBJECT","TRIALINDEX"]):
            subframe = subframe.copy()
            minval = subframe["TIMESTAMP"].min()
            subframe["TRIALTIMESTAMP"] = subframe.apply(lambda row: row["TIMESTAMP"] - minval, axis=1)
            components.append(subframe)
        intermediate = pandas.merge(intermediate, pandas.concat(components))


        #Generating Original Columns:

        #XSERIES
        if average_x in dfcolumns:
            intermediate["XSERIES_ORIG"] = df[average_x]
            HAS_XSERIES = True
        else:
            intermediate["XSERIES_ORIG"] = None
            HAS_XSERIES = False

        if right_x != None and left_x != None:
            intermediate["XSERIES_R_ORIG"] = df[right_x]
            intermediate["XSERIES_L_ORIG"] = df[left_x]
            HAS_LR_XSERIES = True
        else:
            intermediate["XSERIES_R_ORIG"] = None
            intermediate["XSERIES_L_ORIG"] = None
            HAS_LR_XSERIES = False

        if HAS_XSERIES == False and HAS_LR_XSERIES == True:
            intermediate["XSERIES_ORIG"] = df[[right_x, left_x]].mean(axis=1)
            HAS_XSERIES = True

        #YSERIES
        if average_y in dfcolumns:
            intermediate["YSERIES_ORIG"] = df[average_y]
            HAS_YSERIES = True
        else:
            intermediate["YSERIES_ORIG"] = None
            HAS_YSERIES = False

        if right_y != None and left_y != None:
            intermediate["YSERIES_R_ORIG"] = df[right_y]
            intermediate["YSERIES_L_ORIG"] = df[left_y]
            HAS_LR_YSERIES = True
        else:
            intermediate["YSERIES_R_ORIG"] = None
            intermediate["YSERIES_L_ORIG"] = None
            HAS_LR_YSERIES = False

        if HAS_YSERIES == False and HAS_LR_YSERIES == True:
            intermediate["YSERIES_ORIG"] = df[[right_y, left_y]].mean(axis=1)
            HAS_YSERIES = True

        #PUPILDIAMETER
        if average_pupildiameter in dfcolumns:
            intermediate["PUPILDIAMETER_ORIG"] = df[average_pupildiameter]
            HAS_PUPILDIAMETER = True
        else:
            intermediate["PUPILDIAMETER_ORIG"] = None
            HAS_PUPILDIAMETER = False

        if right_pupildiameter != None and left_pupildiameter != None:
            intermediate["PUPILDIAMETER_R_ORIG"] = df[right_pupildiameter]
            intermediate["PUPILDIAMETER_L_ORIG"] = df[left_pupildiameter]
            HAS_LR_PUPILDIAMETER = True
        else:
            intermediate["PUPILDIAMETER_R_ORIG"] = None
            intermediate["PUPILDIAMETER_L_ORIG"] = None
            HAS_LR_PUPILDIAMETER = False

        if HAS_PUPILDIAMETER == False and HAS_LR_PUPILDIAMETER == True:
            intermediate["PUPILDIAMETER_ORIG"] = df[[right_pupildiameter, left_pupildiameter]].mean(axis=1)
            HAS_PUPILDIAMETER = True

        #VALIDITY
        if average_validity in dfcolumns:
            intermediate["VALIDITY_ORIG"] = df[average_validity]
            HAS_VALIDITY = True
        else:
            intermediate["VALIDITY_ORIG"] = None
            HAS_VALIDITY = False

        if right_validity != None and left_validity != None:
            intermediate["VALIDITY_R_ORIG"] = df[right_validity]
            intermediate["VALIDITY_L_ORIG"] = df[left_validity]
            HAS_LR_VALIDITY = True
        else:
            intermediate["VALIDITY_R_ORIG"] = None
            intermediate["VALIDITY_L_ORIG"] = None
            HAS_LR_VALIDITY = False

        if HAS_VALIDITY == False and HAS_LR_VALIDITY == True:
            intermediate["VALIDITY_ORIG"] = df[[right_validity, left_validity]].mean(axis=1)
            HAS_VALIDITY = True

        #Add the onscreen column. Starts as all true.
        intermediate["ONSCREEN_R_ORIG"] = True
        intermediate["ONSCREEN_L_ORIG"] = True
        intermediate["ONSCREEN_ORIG"] = True

        for transfercolumn in transfercolumns:
            intermediate[transfercolumn.upper()] = df[transfercolumn]

        gaze = Gaze(intermediate, screen=screen, accepted_quality=accepted_quality, transfercolumns=[transfercolumn.upper() for transfercolumn in transfercolumns if transfercolumn in list(intermediate.columns.values)])

        return gaze

    @classmethod
    def from_file(cls, csvfile, **kwargs):
        try:
            df = pandas.read_table(cleanPathString(csvfile))
        except:
            print("[ERROR]: File '{0}' could not be found!".format(cleanPathString(csvfile)))
            raise

        try:
            gaze = cls.from_dataframe(df, **kwargs)
        except:
            print("[ERROR]: DataFrame could not be converted to Gaze object!")
            gaze = None
            raise
        return gaze

    def location_is_pixels():
        self.LocationType = "Pixels"

    def location_is_percent():
        self.LocationType = "Percent"

    def set_screen(self, screen):
        self.Screen = screen
        if "XSERIES" in self.LR_Measures and "YSERIES" in self.LR_Measures:
            for suffix in ["_R", "_L"]:
                self.DataFrame["ONSCREEN"+suffix] = self.DataFrame.apply(lambda row: self.Screen.is_inside(x=row["XSERIES"+suffix], y=row["YSERIES"+suffix], locationtype = self.LocationType), axis=1)
            self.__recalculate__("ONSCREEN")
        else:
            self.DataFrame["ONSCREEN"] = self.DataFrame.apply(lambda row: self.Screen.is_inside(x=row["XSERIES"], y=row["YSERIES"], locationtype = self.LocationType), axis=1)
            self.DataFrame["ONSCREEN_R"] = self.DataFrame["ONSCREEN"]
            self.DataFrame["ONSCREEN_L"] = self.DataFrame["ONSCREEN"]

        for measure in self.Measures:
            self.__recalculate__(measure)

    def reset_measures(self):
        for measure in self.Measures:
            self.__recalculate__(measure, reset=True)
        self.__detect_location_type__()
        self.Interpolated = False

    def subset(self, **kwargs):
        #print(GazeSubset(self, **kwargs).DataFrame)
        return GazeSubset(self, **kwargs)

    def pixels_to_percents(self, screen=None):
        if self.Screen == None and screen == None:
            raise TypeError("You must define a screen to use this method.")
            return

        if self.LocationType == "Percent":
            raise TypeError("The Location information is already in percent form.")
            return

        if "XSERIES" in self.LR_Measures and "YSERIES" in self.LR_Measures:
            self.DataFrame[["XSERIES_R","YSERIES_R"]] = self.DataFrame.apply(screen.get_relative_location, xseries="XSERIES_R", yseries="YSERIES_R", axis=1)
            self.DataFrame[["XSERIES_L","YSERIES_L"]] = self.DataFrame.apply(screen.get_relative_location, xseries="XSERIES_L", yseries="YSERIES_L", axis=1)
            self.__recalculate__("XSERIES")
            self.__recalculate__("YSERIES")
            self.LocationType = "Percent"
        elif "XSERIES" in self.Measures and "YSERIES" in self.Measures:
            self.DataFrame[["XSERIES","YSERIES"]] = self.DataFrame.apply(screen.get_relative_location, xseries="XSERIES", yseries="YSERIES", axis=1)
        else:
            raise IOError("Location information not included in Gaze.")

    def flip_axis(self, screen=None, axis=None):
        if self.Screen == None and screen == None and self.LocationType == "Pixels":
            raise TypeError("You must define a screen to use this method, or convert to percent.")
            return

        if axis.upper() in ["X", "WIDTH"] and "XSERIES" in self.LR_Measures:
            if self.LocationType == "Pixels":
                self.DataFrame["XSERIES_R"] = self.Screen.Width - self.DataFrame["XSERIES_R"]
                self.DataFrame["XSERIES_L"] = self.Screen.Width - self.DataFrame["XSERIES_L"]
            else:
                self.DataFrame["XSERIES_R"] = 1 - self.DataFrame["XSERIES_R"]
                self.DataFrame["XSERIES_L"] = 1 - self.DataFrame["XSERIES_L"]
            self.__recalculate__("XSERIES")

        elif axis.upper() in ["X", "WIDTH"] and "XSERIES" in self.Measures:
            if self.LocationType == "Pixels":
                self.DataFrame["XSERIES"] = self.Screen.Width - self.DataFrame["XSERIES"]
            else:
                self.DataFrame["XSERIES"] = 1 - self.DataFrame["XSERIES"]

        elif axis.upper() in ["Y", "HEIGHT"] and "YSERIES" in self.LR_Measures:
            if self.LocationType == "Pixels":
                self.DataFrame["YSERIES_R"] = self.Screen.Width - self.DataFrame["YSERIES_R"]
                self.DataFrame["YSERIES_L"] = self.Screen.Width - self.DataFrame["YSERIES_L"]
            else:
                self.DataFrame["YSERIES_R"] = 1 - self.DataFrame["YSERIES_R"]
                self.DataFrame["YSERIES_L"] = 1 - self.DataFrame["YSERIES_L"]
            self.__recalculate__("YSERIES")

        elif axis.upper() in ["Y", "WIDTH"] and "YSERIES" in self.Measures:
            if self.LocationType == "Pixels":
                self.DataFrame["YSERIES"] = self.Screen.Width - self.DataFrame["YSERIES"]
            else:
                self.DataFrame["YSERIES"] = 1 - self.DataFrame["YSERIES"]

        else:
            raise IOError("Location information not included in Gaze, or an unrecognized axis was specified.")

    def set_quality_range(self, lower=0, upper=100):
        self.AcceptedQuality=(lower,upper)
        for measure in self.Measures:
            self.__recalculate__(measure)

    def sg_hpfilter(self, window_size=15, poly_order=3, deriv=0):
        """Implementation of the Savitzky-Golay filter -- taken from:
        http://www.scipy.org/Cookbook/SavitzkyGolay

        Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
        The Savitzky-Golay filter removes high frequency noise from data.
        It has the advantage of preserving the original shape and
        features of the signal better than other types of filtering
        approaches, such as moving averages techhniques.
        Parameters
        ----------
        y : array_like, shape (N,)
            the values of the time history of the signal.
        window_size : int
            the length of the window. Must be an odd integer number.
        poly_order : int
            the order of the polynomial used in the filtering.
            Must be less than `window_size` - 1.
        deriv: int
            the order of the derivative to compute (default = 0 means only smoothing)"""

        try:
            window_size = np.abs(np.int(window_size))
            order = np.abs(np.int(poly_order))
        except ValueError, msg:
            raise ValueError("window_size and order have to be of type int")
        if window_size % 2 != 1 or window_size < 1:
            raise TypeError("window_size size must be a positive odd number")
        if window_size < poly_order + 2:
            raise TypeError("window_size is too small for the polynomials order")
        order_range = range(poly_order+1)
        half_window = (window_size -1) // 2

        #Generate a list of columns to filter
        serieslist = []
        for measure in ["XSERIES", "YSERIES", "PUPILDIAMETER"]:
            if measure in self.LR_Measures:
                serieslist.extend([measure+"_R", measure+"_L"])
            elif measure in self.Measures:
                serieslist.append(measure)

        #Filter those columns
        for series in serieslist:
            results = []
            for groupid, dataslice in self.DataFrame.groupby(["SUBJECT", "STIMULUS"]):
                y=list(dataslice[series])

                # precompute coefficients
                b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
                m = np.linalg.pinv(b).A[deriv]
                # pad the signal at the extremes with
                # values taken from the signal itself
                firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
                lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
                y = np.concatenate((firstvals, y, lastvals))
                dataslicecopy=dataslice.copy()
                dataslicecopy[series+"_filtered"] = np.convolve( m[::-1], y, mode='valid')
                results.append(dataslicecopy[series+"_filtered"])
            self.DataFrame[series] = pandas.concat(results)

        #Recalculate as necessary
        for measure in self.Measures:
            self.__recalculate__(measure)

    # def trim_to_screen(self, screen=None):
    #     if self.Screen == None and screen == None and self.LocationType == "Pixels":
    #         raise TypeError("You must define a screen to use this method, or convert to percent.")
    #         return
    #
    #     if self.LocationType == "Percent":
    #         self.DataFrame[["XSERIES","YSERIES","PUPILDIAMETER"]] = self.DataFrame.apply(self.Screen.check_boundary, axis=1, locationtype = self.LocationType)

    def interpolate_missing(self, limit=75):
        #Generate a list of columns to filter
        serieslist = []
        for measure in ["XSERIES", "YSERIES", "PUPILDIAMETER"]:
            if measure in self.LR_Measures:
                serieslist.extend([measure+"_R", measure+"_L"])
            elif measure in self.Measures:
                serieslist.append(measure)

        #Filter those series
        for series in serieslist:
            results = []
            for groupid, dataslice in self.DataFrame.groupby(["SUBJECT", "STIMULUS"]):
                dataslice=dataslice.copy()
                dataslice["ORIG_INDEX"] = dataslice.index
                dataslice.index = pandas.to_datetime(dataslice["TRIALTIMESTAMP"], unit="ms")
                dataslice[series] = dataslice[series].interpolate(method="time", limit=limit/2, limit_direction="both")
                dataslice.index = dataslice["ORIG_INDEX"]
                #dataslice.drop("ORIG_INDEX", 1)
                results.append(dataslice[series])
            self.DataFrame[series] = pandas.concat(results)

        self.Interpolated=True

        #Recalculate as necessary
        for measure in self.Measures:
            self.__recalculate__(measure)

    def denoise(self, window = 9, threshold = 2):
        """
        Removes samples in a rolling window meeting a given threshold
        ---------
        window: a odd-numbered int
        """
        if window % 2 == 0 or window < 3 or type(window) != int:
            raise TypeError("window must be an odd integer above 3.")

        window_middle = (window+1)/2
        samples = range(window_middle-window, window-window_middle+1)

        serieslist = []
        for measure in ["XSERIES", "YSERIES", "PUPILDIAMETER"]:
            if measure in self.LR_Measures:
                serieslist.extend([measure+"_R", measure+"_L"])
            elif measure in self.Measures:
                serieslist.append(measure)

        for series in serieslist:
            window_columns = []
            measureseries = self.DataFrame[[series]].copy()
            for offset in samples:
                offsetcolumnname = series+"_"+str(offset)
                window_columns.append(offsetcolumnname)
                measureseries[offsetcolumnname] = measureseries[series].shift(offset)
            stdcolumn = series+"_std"
            #maskcolumn = series+"_mask"
            maskedcolumn = series+"_masked"
            stdseries = measureseries[window_columns].std(axis=1)
            avgseries = measureseries[window_columns].mean(axis=1)
            upperboundseries = avgseries + (stdseries * threshold)
            lowerboundseries = avgseries - (stdseries * threshold)
            workingframe = pandas.DataFrame({"measure":measureseries[series], "lowerbound": lowerboundseries, "upperbound":upperboundseries})
            maskseries = workingframe.apply(lambda row: False if row["lowerbound"] <= row["measure"] <= row["upperbound"] else True, axis=1)
            maskedseries = measureseries[series].mask(maskseries)
            #resultframe = pandas.DataFrame({series:measureseries[series], stdcolumn:stdseries, maskedcolumn:maskedseries})
            self.DataFrame[series] = maskedseries

        for measure in self.Measures:
            self.__recalculate__(measure)

    def __compute_velocity__(self, row):
        if None in row.unique():
            return None
        else:
            #orig_degree = math.degrees(math.tan((row["YSERIES"]-row["YSERIES-PREVIOUS"]) / (row["XSERIES"]-row["XSERIES-PREVIOUS"])))
            #next_degree = math.degrees(math.tan((row["YSERIES-NEXT"]-row["YSERIES"]) / (row["XSERIES-NEXT"]-row["XSERIES"])))
            return math.fabs(math.sqrt(math.pow(row["YSERIES-NEXT"]-row["YSERIES"],2)+math.pow(row["XSERIES-NEXT"]-row["XSERIES"],2)) / (row["TRIALTIMESTAMP-NEXT"] - row["TRIALTIMESTAMP"]))
            #delta_time = row["TRIALTIMESTAMP-NEXT"] - row["TRIALTIMESTAMP"]
            #return math.fabs(delta_space / delta_time)


    def find_fixations(self, velocity_threshold=.20, window=11):
        results = []
        for groupid, dataslice in self.DataFrame.groupby(["SUBJECT", "STIMULUS"]):
            dataslice=dataslice[["XSERIES", "YSERIES", "TRIALTIMESTAMP"]].copy()
            #dataslice["XSERIES-PREVIOUS"] = dataslice["XSERIES"].shift(1)
            #dataslice["YSERIES-PREVIOUS"] = dataslice["YSERIES"].shift(1)
            dataslice["TRIALTIMESTAMP-NEXT"] = dataslice["TRIALTIMESTAMP"].shift(1)
            dataslice["XSERIES-NEXT"] = dataslice["XSERIES"].shift(-1)
            dataslice["YSERIES-NEXT"] = dataslice["YSERIES"].shift(-1)
            dataslice["VELOCITY"] = dataslice.apply(lambda row: self.__compute_velocity__(row), axis=1)


            #Smooth velocity via window
            if window % 2 == 0 or window < 3 or type(window) != int:
                raise TypeError("window must be an odd integer above 3.")

            window_middle = (window+1)/2
            samples = range(window_middle-window, window-window_middle+1)
            window_columns=[]
            for offset in samples:
                offsetcolumnname = "VELOCITY_"+str(offset)
                window_columns.append(offsetcolumnname)
                dataslice[offsetcolumnname] = dataslice["VELOCITY"].shift(offset)
            dataslice["VELOCITY_SMOOTHED"] = dataslice[window_columns].mean(axis=1)
            dataslice["VELOCITY_SMOOTHED"] = (dataslice["VELOCITY_SMOOTHED"]/dataslice["VELOCITY_SMOOTHED"].max())
            dataslice["STABILITY"] = dataslice["VELOCITY_SMOOTHED"].apply(lambda value: False if value >= velocity_threshold else True)


            dataslice[["XSERIES","YSERIES","VELOCITY","VELOCITY_SMOOTHED","STABILITY"]].plot()
            plt.show()
            results.append(dataslice["VELOCITY"])
        sys.exit()
        self.DataFrame["VELOCITY"] = pandas.concat(results)


    def plot(self, **kwargs):
        if self.Screen == None and screen == None:
            raise TypeError("You must define a screen to use this method")
            return
        if len(kwargs.keys()) != 0:
            #print(kwargs)
            self.Screen.plot(self.subset(**kwargs), **kwargs)

    def summary(self, summary_type, by_stimulus=True, by_epoch=True, format="long"):
        if summary_type not in ["percent_time", "percent_fixation", "time_to_fixation"]:
            raise TypeError("Summary type is not valid.")
        groups = ["SUBJECT"]
        summarylist = []
        if by_epoch == True and len(self.Epochs) > 0:
            by_epoch = True
        else:
            by_epoch = False
            print("No epochs provided. Disregarding 'by_epoch' flag.")

        if by_stimulus == True and len(self.Epochs) > 0:
            by_stimulus = True
            groups.append("STIMULUS")
        else:
            by_stimulus = False
            groups.append("TRIALINDEX")
            print("No aois provided. Disregarding 'by_aoi' flag.")

        for index, dataslice in self.DataFrame.groupby(groups):
            if by_stimulus == False:
                aois = ["ANYAOI"]

        pass

    def image_summary(self, outfile, epochs=None):
        subjectcount = len(self.DataFrame["SUBJECT"].unique())
        if subjectcount <= 3:
            opacity = .75
        else:
            opacity = 3.0/subjectcount
        colors = ["b","g","r","c","m","y","k"]
        if epochs==None:
            epochs=self.Epochs
        summaryframe = self.DataFrame[["SUBJECT","TRIALINDEX","TRIALTIMESTAMP","TIMESTAMP","ANYAOI"]+self.Epochs+self.UtilityColumns].copy()
        relevanttrials = []
        for index, dataslice in summaryframe.groupby(["SUBJECT","TRIALINDEX"]):
            if True in dataslice["ANYAOI"].unique():
                relevanttrials.append(dataslice)
        if len(relevanttrials) != 0:
            summaryframe = pandas.concat(relevanttrials)
        else:
            raise IOError("No AOIs were observed!")

        fig = plt.figure()
        fig.canvas.set_window_title("Looks in Any AOI over Time")
        plt.title("Looks in Any AOI over Time")
        plt.xlabel("Time (ms)")
        plt.ylabel("Percentage in Any AOI")
        plt.ylim((0,100))
        aoicolumns = []
        if len(epochs) < 1:
            aoicolumns.append("ANYAOI")
        else:
            for epoch in self.Epochs:
                aoicolumns.append([epoch,"ANYAOI_{0}".format(epoch)])
                #summaryframe["ANYAOI_{0}".format(epoch)] = summaryframe["ANYAOI"].mask(summaryframe[epoch].apply(lambda value: True if value <= 0.5 else False))

        #Merge across subject
        #subjectframes = []
        for index, dataslice in summaryframe.groupby(["SUBJECT"]):
            subjtrialframe = dataslice.copy()
            subjtrialframe.index = pandas.to_datetime(subjtrialframe["TRIALTIMESTAMP"], unit="ms")
            subjtrialframe = subjtrialframe.resample("50ms")
            subjtrialframe.index = subjtrialframe["TRIALTIMESTAMP"]
            for aoiindex, aoicolumnset in enumerate(aoicolumns):
                subjtrialframe["ANYAOI_{0}".format(aoicolumnset[0])] = subjtrialframe["ANYAOI"].mask(subjtrialframe[aoicolumnset[0]].apply(lambda value: True if value <= 0.5 else False))
                plt.plot(subjtrialframe["TRIALTIMESTAMP"], subjtrialframe[aoicolumnset[1]]*100, color=colors[aoiindex], alpha=opacity, linewidth=2, label=aoicolumnset[0])

        #print(summaryframe)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))
        #plt.show()
        plt.savefig(outfile,bbox_inches='tight')

    def add_timing(self, timingobject):
        self.MetaData = list(set(self.MetaData+timingobject.MetaData))
        mergedframe = pandas.merge(self.DataFrame, timingobject.DataFrame)
        mergedframe.to_csv("/scratch/schoen/addedtiming.csv")
        if len(mergedframe) < 1:
            raise IOError("Column values do not match up between Gaze and Timing")
        mergedframe['TIMESTAMP_SHIFT'] = mergedframe['TIMESTAMP'].shift(-1)
        mergedframe['TRIALINDEX_SHIFT'] = mergedframe['TRIALINDEX'].shift(-1)
        to_drop = ['TIMESTAMP_SHIFT', 'TRIALINDEX_SHIFT']
        for epoch in timingobject.Epochs:
            epochonset=epoch+".ONSET"
            epochoffset=epoch+".OFFSET"
            print 'Processing Epoch {0}'.format(epoch)
            self.Epochs.append(epoch)
            self.UtilityColumns.extend([epochonset,epochoffset])
            mergedframe[str(epoch)] = np.apply_along_axis(lambda row: self.__row_epoch_inside__(row[0],
                                                                                                row[1],
                                                                                                row[2],
                                                                                                row[3],
                                                                                                epoch_onset=row[4],
                                                                                                epoch_offset=row[5]),
                                                                                            axis=1,
                                                                                            arr=mergedframe.as_matrix(columns=["TIMESTAMP",
                                                                                                                                  "TIMESTAMP_SHIFT",
                                                                                                                                  "TRIALINDEX",
                                                                                                                                  "TRIALINDEX_SHIFT",
                                                                                                                                  epochonset,
                                                                                                                                  epochoffset]))
            #mergedframe[str(epoch)] = mergedframe[["TIMESTAMP","TIMESTAMP_SHIFT","TRIALINDEX","TRIALINDEX_SHIFT",epoch+".ONSET",epoch+".OFFSET"]].apply(lambda row: self.__row_epoch_inside__(row["TIMESTAMP"], row["TIMESTAMP_SHIFT"], row["TRIALINDEX"], row["TRIALINDEX_SHIFT"], epoch_onset = row[epoch+".ONSET"], epoch_offset = row[epoch+".OFFSET"]), axis=1)

        mergedframe.drop(to_drop, axis=1, inplace=True)
        mergedframe["TRIALTIMESTAMP"] = mergedframe["TIMESTAMP"] - mergedframe["TRIAL.ONSET"]
        self.DataFrame = mergedframe

    def add_aoi(self, aoicollection, byrank = False, aoicode="STIMULUS"):
        print("Processing AOI {0}".format(aoicollection.Name))
        self.DataFrame[str(aoicollection.Name)] = self.DataFrame[[aoicode,"XSERIES","YSERIES"]].apply(lambda row: aoicollection.__row_inside__(x=row["XSERIES"],y=row["YSERIES"], aoicode=row[aoicode], byrank=False), axis=1)
        self.AOIs.append(aoicollection.Name)
        self.DataFrame["ANYAOI"] = np.apply_along_axis(lambda row: True if row[0] == True or row[1] == True else False, axis=1, arr=self.DataFrame.as_matrix(columns=[str(aoicollection.Name), "ANYAOI"]))

        if byrank != False:
            for rank in aoicollection.Ranks:
                self.DataFrame[str(aoicollection.Name)+"_"+str(rank)] = self.DataFrame[[aoicode,"XSERIES","YSERIES"]].apply(lambda row: aoicollection.__row_inside__(x=row["XSERIES"],y=row["YSERIES"], aoicode=row[aoicode], byrank=rank), axis=1)
                self.AOIs.append(str(aoicollection.Name)+"_"+str(rank))
                self.DataFrame["ANYAOI"] = np.apply_along_axis(lambda row: True if row[0] == True or row[1] == True else False, axis=1, arr=self.DataFrame.as_matrix(columns=[str(aoicollection.Name)+"_"+str(rank), "ANYAOI"]))

        self.AOIs = remove_duplicates(self.AOIs)

    def to_csv(self, outfile, allaois=False, concise=True):
        if allaois == False:
            aois = ["ANYAOI"]
        else:
            aois = self.AOIs
            aois.remove("ANYAOI")
            aois = ["ANYAOI"] + aois
        if concise == False:
            datacolumns = self.VisibleColumns + self.UtilityColumns
        else:
            datacolumns = self.VisibleColumns
        includedheaders = remove_duplicates(datacolumns + self.MetaData + self.Epochs + aois)
        writecsv = self.DataFrame[includedheaders]
        writecsv.to_csv(outfile, index=False)

class GazeAnimation(object):
    """
    The GazeAnimation class
    Gaze specifically geared towards animation.
    """
    def __init__(self, screen, df, locationtype, dot, xgaze, ygaze, pgaze, aoicollection):
        self.Screen = screen
        self.DataFrame = df
        self.LocationType = locationtype
        self.Dot = dot
        self.Xgaze = xgaze
        self.Ygaze = ygaze
        self.Pgaze = pgaze
        self.AOICollection = aoicollection
        if self.LocationType == "Pixels":
            self.DisplayCoordinates = zip(list(self.DataFrame['XSERIES']), list(self.DataFrame['YSERIES']), list(self.DataFrame['PUPILDIAMETER']))
        else:
            self.DisplayCoordinates = zip(list(self.DataFrame['XSERIES']*self.Screen.Width), list(self.DataFrame['YSERIES']*self.Screen.Height), list(self.DataFrame['PUPILDIAMETER']))
        self.Dot.center = (-100, -100)
        self.Xgaze.set_y(-100)
        self.Ygaze.set_x(-100)
        self.Pgaze.set_x(-100)

    def animate_set(self):
        self.Dot.center = (-100, -100)
        self.Xgaze.set_y(-100)
        self.Ygaze.set_x(-100)
        self.Pgaze.set_x(-100)
        return (self.Dot, self.Xgaze, self.Ygaze, self.Pgaze)

    def animate_gaze(self, i):
        self.Xgaze.set_y(i)
        self.Ygaze.set_x(i)
        self.Pgaze.set_x(i)
        #self.TimeBar.set_width(self.Screen.Width * (float(i) / float(self.Frames)))
        if self.DisplayCoordinates[i][0] != None and self.DisplayCoordinates[i][1] != None:
            #Set the dot's radius based on the pupil diameter, if possible
            if self.DisplayCoordinates[i][2] != None:
                self.Dot.set_radius(3*self.DisplayCoordinates[i][2])
            else:
                self.Dot.set_radius(10)

            #Set the dot's center
            self.Dot.center = (self.DisplayCoordinates[i][0],self.DisplayCoordinates[i][1])

            if self.AOICollection.is_inside(x=self.DisplayCoordinates[i][0]/self.Screen.Width, y=self.DisplayCoordinates[i][1]/self.Screen.Height):
                self.Dot.set_facecolor('green')
                self.Xgaze.set_facecolor('green')
                self.Ygaze.set_facecolor('green')
                self.Pgaze.set_facecolor('green')
            else:
                self.Dot.set_facecolor('red')
                self.Xgaze.set_facecolor('red')
                self.Ygaze.set_facecolor('red')
                self.Pgaze.set_facecolor('red')
        else:
            self.Dot.center = (-100, -100)
        return (self.Dot, self.Xgaze, self.Ygaze, self.Pgaze)

class Screen(object):
    """
    The screen class.
    Provides info about the screen settings
    """

    def __init__(self, width = 800, height = 600):
        self.Width = width
        self.Height = height
        self.Area = width * height

    def is_inside(self, x = None, y = None, locationtype = "Pixels"):
        if x == None or y == None:
            return False
        elif locationtype == "Pixels" and 0 <= x <= self.Width and 0 <= y <= self.Height:
            return True
        elif locationtype == "Percent" and 0 <= x <= 1 and 0 <= x <= 1:
            return True
        else:
            return False

    def row_inside(self, row, x = None, y = None, locationtype = "Pixels"):
        return self.is_inside(x=row[x], y=row[y], locationtype=locationtype)

    def get_relative_location(self, row, xseries="XSERIES", yseries="YSERIES"):
        xrel = float(row[xseries])/self.Width
        yrel = float(row[yseries])/self.Height
        return pandas.Series([xrel, yrel])

    def __str__(self):
        string = 'Screen with parameters:\n    Width: {0}\n    Height: {1}\n    Area: {2}'.format(self.Width, self.Height, self.Area)
        return string

    def plot(self, gaze, aoicollection = None, background = None, **kwargs):
        #print(gaze.DataFrame)
        #print(gaze.DataFrame["SUBJECT"].unique())
        #print(gaze.DataFrame["STIMULUS"].unique())
        blackscreen = np.zeros([self.Height,self.Width,3],dtype=np.uint8)
        blackscreen.fill(1)
        greyscreen = np.zeros([self.Height,self.Width,3],dtype=np.uint8)
        greyscreen.fill(100)

        if aoicollection == None:
            aoicollection = AOICollection()
        fig = plt.figure(figsize=(self.Width/66,self.Height/66))
        #fig.canvas.set_width(self.Width*2)
        #fig.canvas.set_height(self.Height*2)
        fig.canvas.set_window_title(aoicollection.Name)
        plt.subplot(221)
        if background != None:
            try:
                image = Image.open(cleanPathString(background)).resize((self.Width, self.Height)).transpose(Image.FLIP_TOP_BOTTOM)
                print("Loaded image.")
            except:
                print("Image not loaded.")
                image = greyscreen
        else:
            image = greyscreen

        if aoicollection != None:
            try:
                aoioverlay = aoicollection.to_image(self)
                aoialpha = 0.5
                print("Loaded AOI.")
            except:
                print("AOI not loaded.")
                aoioverlay = blackscreen
                aoialpha = 0.0
        else:
            aoioverlay = blackscreen
            aoialpha = 0.0

        plt.imshow(image, origin='lower')
        plt.imshow(aoioverlay, alpha=aoialpha, origin='lower')
        imageaxis = plt.gca()
        gazedot = ptc.Circle((self.Width / 2, self.Height / 2), 10, fc='red', alpha=0.75)
        gazedot.set_edgecolor('none')

        #serieslength = (gaze.DataFrame.index-gaze.DataFrame.index[0]).max()
        xgazeline = ptc.Rectangle((0, 0), self.Width, 10, fc='red', alpha=0.5)
        xgazeline.set_edgecolor("none")
        ygazeline = ptc.Rectangle((0, 0), 10, self.Height, fc='red', alpha=0.5)
        ygazeline.set_edgecolor("none")
        pgazeline = ptc.Rectangle((0, gaze.DataFrame["PUPILDIAMETER"].min()-1), 10, gaze.DataFrame["PUPILDIAMETER"].max()+1, fc='red', alpha=0.5)
        pgazeline.set_edgecolor("none")

        imageaxis.add_patch(gazedot)
        plt.title("Image")

        plt.subplot(222)
        #print(gaze.DataFrame)
        plt.plot(list(gaze.DataFrame.index-gaze.DataFrame.index[0]), list(gaze.DataFrame["YSERIES"]*self.Height), linewidth=2, color="blue")
        plt.plot(list(gaze.DataFrame.index-gaze.DataFrame.index[0]), list(gaze.DataFrame["YSERIES_ORIG"]*self.Height), color="orange", alpha=0.2)
        ygazeaxis = plt.gca()
        ygazeaxis.add_patch(ygazeline)
        ygazeaxis.set_ylim([0,self.Height])
        plt.title("Y-Axis Gaze")

        plt.subplot(223)
        plt.plot(list(gaze.DataFrame["XSERIES"]*self.Width), list(gaze.DataFrame.index-gaze.DataFrame.index[0]), linewidth=2, color="blue")
        plt.plot(list(gaze.DataFrame["XSERIES_ORIG"]*self.Width),list(gaze.DataFrame.index-gaze.DataFrame.index[0]), color="orange", alpha=0.2)
        xgazeaxis = plt.gca()
        xgazeaxis.invert_yaxis()
        xgazeaxis.add_patch(xgazeline)
        xgazeaxis.set_xlim([0,self.Width])
        plt.title("X-Axis Gaze")

        plt.subplot(224)
        plt.plot(list(gaze.DataFrame.index-gaze.DataFrame.index[0]), list(gaze.DataFrame["PUPILDIAMETER"]), linewidth=2, color="blue")
        plt.plot(list(gaze.DataFrame.index-gaze.DataFrame.index[0]), list(gaze.DataFrame["PUPILDIAMETER_ORIG"]), color="orange", alpha=0.2)
        pupilgazeaxis = plt.gca()
        pupilgazeaxis.add_patch(pgazeline)
        pupilgazeaxis.set_ylim([gaze.DataFrame["PUPILDIAMETER"].min()-1,gaze.DataFrame["PUPILDIAMETER"].max()+1])
        plt.title("Pupil Diameter")

        fig.tight_layout()

        if gaze.Frames > 0:
            print('Adding the gaze to the image.')
            gazeanimation = GazeAnimation(self, gaze.DataFrame[["XSERIES", "YSERIES", "PUPILDIAMETER"]], gaze.LocationType, gazedot, xgazeline, ygazeline, pgazeline, aoicollection)
            print("Drawing")
            gaze_animation = animation.FuncAnimation(fig, gazeanimation.animate_gaze, init_func=gazeanimation.animate_set, frames=gaze.Frames, interval= 1 / gaze.Hertz, blit=True)
        else:
            print('Quality gaze data not available for this image.')
            gazedot.center = (-100, -100)

        plt.show()

class AOICollection(object):
    """
    The AOICollection class.
    Defines a collection of AOIs, likely all relating to a given stimulus.
    """

    def __init__(self, name = 'Group', obt = None, json = None, screen = Screen()):
        self.Ranks = set([])
        self.Contents = []
        self.updateRanks()
        self.Name = name
        if obt != None:
            self.read_obt(obt, screen)
        if json != None:
            self.read_json(json)

    def updateRanks(self):
        ranklist = []
        for aoi in self.Contents:
            ranklist.append(aoi.Rank)

        self.Ranks = set(ranklist)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for i in range(0, len(self.Contents)):
            self.Contents.remove(self.Contents[0])

    def __str__(self):
        if len(self.Contents) == 0:
            string = 'AOI Collection:\n\tEmpty Collection'
        else:
            string = '\n'.join(['AOI Collection:'] + [ x.__str__() for x in self.Contents ])
        return string

    def is_inside(self, x = None, y = None):
        inside = False
        for AOI in self.Contents:
            if AOI.is_inside(x=x, y=y):
                inside = True
        return inside

    def __row_inside__(self, x = None, y = None, aoicode = None, byrank=False):
        #print(self)
        if RepresentsInt(aoicode):
            aoicode = int(aoicode)
        #print(x, y, aoicode)
        if str(aoicode) != str(self.Name):
            return False
        elif byrank == False:
            return self.is_inside(x, y)
        else:
            return self.get_aoi_by_rank(byrank).is_inside(x, y)

    def get_aoi_by_rank(rank):
        #Returns the first aoi with the given rank
        for aoi in self.Contents:
            if aoi.Rank == rank:
                return aoi


    def add_AOI(self, AOI):
        if not hasattr(AOI, 'Type'):
            print 'You are attempting to add a non-AOI object to the AOICollection.'
            return False
        self.Contents.append(AOI)
        self.updateRanks()
        return True


    def read_obt(self, path, screen):
        rawAOIs = []
        if exists(path):
            with open(cleanPathString(path)) as currentFileName:
                lines = currentFileName.readlines()
                for line in lines:
                    linecontents = []
                    dirtyContent = re.split(', |  |=', line)
                    content = map(int, [ x for x in dirtyContent if RepresentsInt(x) ])
                    AOIinfo = {}
                    if len(content) > 1:
                        rawAOIs.append(content)

            rank = 1
            for rawAOI in rawAOIs:
                if rawAOI[0] == 1:
                    AOI = Rectangle_AOI(self, Rank=rank, Xmin=rawAOI[1] / float(screen.Width), Xmax=rawAOI[3] / float(screen.Width), Ymax=1 - rawAOI[2] / float(screen.Height), Ymin=1 - rawAOI[4] / float(screen.Height))
                    if AOI.Area <= 0.95:
                        rank += 1
                        self.add_AOI(AOI)
                elif rawAOI[0] == 2:
                    AOI = Ellipse_AOI(self, Rank=rank, Xcenter=rawAOI[1] / float(screen.Width), Xradius=rawAOI[3] / float(screen.Width), Ycenter=1 - float(rawAOI[2]) / screen.Height, Yradius=rawAOI[4] / float(screen.Height))
                    rank += 1
                    self.add_AOI(AOI)

        else:
            raise IOError('Could not read file from {0}'.format(path))

    def read_json(self, path):
        try:
            with open(cleanPathString(path), 'r+') as infile:
                aoisinfo = byteify(json.load(infile))
                infile.seek(0)
        except:
            raise IOError('Could not read file from {0}'.format(path))

        for aoiinfo in aoisinfo:
            if aoiinfo['Type'] == 'Rectangle':
                self.add_AOI(Rectangle_AOI(self, **aoiinfo))
            elif aoiinfo['Type'] == 'Ellipse':
                self.add_AOI(Ellipse_AOI(self, **aoiinfo))

    def to_obt(self, path, screen):
        contents = ['[Objects]', 'Object01=1, 0, 0, {0}, {1}  O'.format(screen.Width, screen.Height)]
        for index, aoi in enumerate(self.Contents):
            values = [aoi.Rank + 1, aoi.TypeCode]
            if aoi.Type == 'Rectangle':
                values.extend([int(aoi.Xmin * screen.Width),
                 int((1 - aoi.Ymax) * screen.Height),
                 int(aoi.Xmax * screen.Width),
                 int((1 - aoi.Ymin) * screen.Height)])
            elif aoi.Type == 'Ellipse':
                values.extend([int(aoi.Xcenter * screen.Width),
                 int((1 - aoi.Ycenter) * screen.Height),
                 int(aoi.Xradius * screen.Width),
                 int(aoi.Yradius * screen.Height)])
            else:
                values.extend([int(aoi.Xmin * screen.Width),
                 int(aoi.Ymin * screen.Height),
                 int(aoi.Xmax * screen.Width),
                 int(aoi.Ymax * screen.Height)])
            contents.append('Object{0:02d}={1}, {2}, {3}, {4}, {5}  Object {0}'.format(*values))

        if len(self.Contents) < 15:
            for emptyval in range(len(self.Contents) + 2, 17):
                contents.append('Object{0:02d}=0'.format(emptyval))

        obt = '\n'.join(contents)
        try:
            with open(path, 'w') as outfile:
                outfile.write('{0}\n'.format(obt))
            return True
        except:
            raise IOError('Could not write file to {0}'.format(path))

    def to_json(self, path):
        try:
            contents = []
            for AOI in self.Contents:
                contents.append(AOI.__dict__())

            with open(path, 'w') as outfile:
                json.dump(contents, outfile, sort_keys=True, indent=4, ensure_ascii=False)
            return True
        except:
            raise IOError('Could not write file to {0}'.format(path))

    def to_image(self, screen):
        image = np.zeros([screen.Height,screen.Width,3],dtype=np.uint8)
        image.fill(1)
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        for aoi in self.Contents:
            if aoi.Type == "Rectangle":
                draw.rectangle([aoi.Xmin*screen.Width, aoi.Ymin*screen.Height, aoi.Xmax*screen.Width, aoi.Ymax*screen.Height], fill='#ffffff')
            if aoi.Type == "Ellipse":
                draw.ellipse([aoi.Xmin*screen.Width, aoi.Ymin*screen.Height, aoi.Xmax*screen.Width, aoi.Ymax*screen.Height], fill='#ffffff')

        return image
        # fig = plt.figure(frameon=False)
        #
        # fig.set_dpi(100)
        # fig.set_size_inches(screen.Width / 100, screen.Height / 100)
        # ax = fig.add_axes([0,0,1,1])
        # ax.axis('off')
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        # ax.set_xlim(0, 1)
        # ax.set_ylim(0, 1)
        # ax.add_artist(ptc.Rectangle((0, 0), 1, 1, alpha=1, color='#000000'))
        # for aoi in self.Contents:
        #     if aoi.Type == 'Rectangle':
        #         ax.add_artist(ptc.Rectangle((aoi.Xmin, aoi.Ymin), aoi.Xmax - aoi.Xmin, aoi.Ymax - aoi.Ymin, alpha=1, color='#ffffff'))
        #     elif aoi.Type == 'Ellipse':
        #         ax.add_artist(ptc.Ellipse((aoi.Xcenter, aoi.Ycenter), aoi.Xradius * 2, aoi.Yradius * 2, alpha=1, color='#ffffff'))
        # fig.
        #data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        #data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data

    def write_image(self, path, screen):
        data = self.to_image(screen)
        try:
            #fig.savefig(path, bbox_inches=None, origin='lower', extent=[0,1,0,1])
            misc.imsave(path, data)
        except:
            raise IOError('Could not write file to {0}'.format(path))

class AOI(object):
    """
    The AOI class.
    Defines a region in space.
    Allows you to see if coordinates occur in that space
    """

    def __init__(self, AOICollection, Rank = 1, Type = 'None', Typecode = '0', **kwargs):
        self.ParentCollection = AOICollection
        self.Rank = Rank
        self.Type = Type
        self.TypeCode = Typecode
        self.Xmin = None
        self.Xmax = None
        self.Xcenter = None
        self.Ymin = None
        self.Ymax = None
        self.Ycenter = None
        self.Area = None

    def is_inside(self, x = None, y = None):
        return False

    def __str__(self):
        string = 'ParentCollection: {0}\n  Rank:   {1}\n  Type:   {2}\n  Top:    {3}\n  Bottom: {4}\n  Left:   {5}\n  Right:  {6}'.format(self.ParentCollection.Name, self.Rank, self.Type, self.Ymax, self.Ymin, self.Xmin, self.Xmax)
        return string

    def __dict__(self):
        aoi = {'ParentCollection': self.ParentCollection.Name,
         'Rank': self.Rank,
         'Type': self.Type,
         'TypeCode': self.TypeCode,
         'Xmin': self.Xmin,
         'Xmax': self.Xmax,
         'Xcenter': self.Xcenter,
         'Ymin': self.Ymin,
         'Ymax': self.Ymax,
         'Ycenter': self.Ycenter,
         'Area': self.Area}
        return aoi

    def __repr__(self):
        return self.__str__()

class Rectangle_AOI(AOI):

    def __init__(self, AOICollection, Rank = 1, Type = 'Rectangle', Typecode = '1', Xmin = None, Xmax = None, Xcenter = None, Ymin = None, Ymax = None, Ycenter = None, **kwargs):
        AOI.__init__(self, AOICollection, Rank=Rank, Type=Type, Typecode=Typecode)
        if Xmin != None and Xmax != None:
            self.Xmin = Xmin
            self.Xmax = Xmax
        elif Xcenter != None:
            Xbound = None
            for Xvalue in [Xmin, Xmax]:
                if Xvalue != None:
                    Xbound = Xvalue

            if Xbound != None:
                Xdiff = math.fabs(Xcenter - Xbound)
                self.Xmax = Xcenter + Xdiff
                self.Xmin = Xcenter - Xdiff
        if Ymin != None and Ymax != None:
            self.Ymin = Ymin
            self.Ymax = Ymax
        elif Ycenter != None:
            Ybound = None
            for Yvalue in [Ymin, Ymax]:
                if Yvalue != None:
                    Ybound = Yvalue

            if Ybound != None:
                Ydiff = math.fabs(Ycenter - Ybound)
                self.Ymax = Ycenter + Ydiff
                self.Ymin = Ycenter - Ydiff
        if self.Xmin != None and self.Xmax != None and self.Ymin != None and self.Ymax != None:
            self.Xcenter = (self.Xmin + self.Xmax) / 2
            self.Ycenter = (self.Ymin + self.Ymax) / 2
            self.Area = (self.Xmax - self.Xmin) * (self.Ymax - self.Ymin)
        else:
            raise ValueError('You did not supply enough information to determine the dimensions of the AOI.')

    def is_inside(self, x = None, y = None):
        if x == None or y == None:
            return False
        elif self.Xmin <= x <= self.Xmax and self.Ymin <= y <= self.Ymax and 0 <= y <= 1 and 0 <= x <= 1:
            return True
        else:
            return False

class Ellipse_AOI(AOI):

    def __init__(self, AOICollection, Rank = 1, Type = 'Ellipse', Typecode = '2', Xradius = None, Yradius = None, Xmin = None, Xmax = None, Xcenter = None, Ymin = None, Ymax = None, Ycenter = None, **kwargs):
        AOI.__init__(self, AOICollection, Rank=Rank, Type=Type, Typecode=Typecode)
        if Xmin != None and Xmax != None:
            self.Xmin = Xmin
            self.Xmax = Xmax
        elif Xcenter != None:
            Xbound = None
            if Xradius != None:
                Xbound = Xcenter + Xradius
            else:
                for Xvalue in [Xmin, Xmax]:
                    if Xvalue != None:
                        Xbound = Xvalue

            if Xbound != None:
                Xdiff = math.fabs(Xcenter - Xbound)
                self.Xmax = Xcenter + Xdiff
                self.Xmin = Xcenter - Xdiff
        if Ymin != None and Ymax != None:
            self.Ymin = Ymin
            self.Ymax = Ymax
        elif Ycenter != None:
            Ybound = None
            if Yradius != None:
                Ybound = Ycenter + Yradius
            else:
                for Yvalue in [Ymin, Ymax]:
                    if Yvalue != None:
                        Ybound = Yvalue

            if Ybound != None:
                Ydiff = math.fabs(Ycenter - Ybound)
                self.Ymax = Ycenter + Ydiff
                self.Ymin = Ycenter - Ydiff
        if self.Xmin != None and self.Xmax != None and self.Ymin != None and self.Ymax != None:
            self.Xcenter = (self.Xmin + self.Xmax) / 2
            self.Ycenter = (self.Ymin + self.Ymax) / 2
            self.Xradius = self.Xmax - self.Xcenter
            self.Yradius = self.Ymax - self.Ycenter
            self.Area = self.Xradius * self.Yradius * math.pi
        else:
            raise ValueError('You did not supply enough information to determine the dimensions of the AOI.')

    def is_inside(self, x = None, y = None):
        if x == None or y == None:
            return False
        elif (x - self.Xcenter) * (x - self.Xcenter) / (self.Xradius * self.Xradius) + (y - self.Ycenter) * (y - self.Ycenter) / (self.Yradius * self.Yradius) <= 1 and 0 <= y <= 1 and 0 <= x <= 1:
            return True
        else:
            return False
