import math, os, re, sys, json, logging, datetime
import numpy as np
import pandas
from matplotlib import pyplot as plt
from matplotlib import patches as ptc
from PIL import Image
from matplotlib import animation

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


class Timing(object):

    def __init__(self, timingframe, timingunits, stimuluscolumn):
        dfdict = {}
        self.Windows = []
        self.Params = ['STIMULUS']
        for column in timingframe.columns.values:
            if column == stimuluscolumn:
                dfdict['STIMULUS'] = timingframe[column].apply(stringify)
            elif len(column.upper().split('.')) > 1:
                if column.upper().split('.')[1] in ('OFFSET', 'ONSET'):
                    timingframe[column] = timingframe[column].apply(floatify)
                    if timingunits == 'seconds':
                        timingframe[column] = timingframe[column] / 1000
                    dfdict[column.upper()] = timingframe[column]
                    prefix = column.upper().split('.')[0]
                    if prefix not in self.Windows:
                        self.Windows.append(prefix)
                else:
                    dfdict[column.upper()] = timingframe[column]
                    self.Params.append(column.upper())
            else:
                dfdict[column.upper()] = timingframe[column]
                self.Params.append(column.upper())

        self.Series = pandas.DataFrame(dfdict).set_index('STIMULUS', drop=False)
        for window in self.Windows:
            if window + '.ONSET' in list(self.Series.columns.values):
                self.Series[str(window) + '.ONSET'] = self.Series[str(window) + '.ONSET'].apply(self.__apply_clean_time, end='ONSET')
            else:
                self.Series[str(window) + '.ONSET'] = zero_datetime()
            if window + '.OFFSET' in list(self.Series.columns.values):
                self.Series[str(window) + '.OFFSET'] = self.Series[str(window) + '.OFFSET'].apply(self.__apply_clean_time, end='OFFSET')
            else:
                self.Series[str(window) + '.OFFSET'] = max_datetime()

    def __apply_clean_time(self, time, end):
        if end == 'OFFSET':
            default = 10000000000000
        else:
            default = 0
        if not isinstance(floatify(time), float):
            time = default
        return time


class Gaze(object):

    def __init__(self, gazeframe, screen, timing = None, hertz = 120, yflip = True, xflip = False):
        self.Frames = len(gazeframe.index)
        self.AvailableWidth = screen.Width
        self.AvailableHeight = screen.Height
        self.Series = gazeframe.copy()
        self.reorganize_columns()
        self.zero_timestamp()
        self.assemble_index()
        self.ExcessColumns = ['TIMESTAMP']
        if isinstance(timing, Timing):
            self.add_timing(timing)
        self.filter_negatives()
        if yflip == True:
            self.flip_axis('Y')
        if xflip == True:
            self.flip_axis('X')
        self.average_eyes()
        self.Series['ScreenX'] = self.Series['XSERIES'] * self.AvailableWidth
        self.Series['ScreenY'] = self.Series['YSERIES'] * self.AvailableHeight
        self.ExcessColumns.extend(['ScreenX', 'ScreenY', 'EXPSAMPLE'])
        self.Hertz = hertz
        self.Duration = self.Frames / self.Hertz

    def reorganize_columns(self):
        columns = list(self.Series.columns.values)
        self.Series['EXPTIMESTAMP'] = self.Series['TIMESTAMP']
        self.Series['TRIALTIMESTAMP'] = None
        standardcolumns = ['STIMULUS',
         'TRIALTIMESTAMP',
         'EXPTIMESTAMP',
         'RXEYE',
         'LXEYE',
         'RYEYE',
         'LYEYE',
         'RVALIDITY',
         'LVALIDITY',
         'RPUPILDIAMETER',
         'LPUPILDIAMETER']
        nonstandardcolumns = [ item for item in columns if item not in standardcolumns ]
        columns = standardcolumns + nonstandardcolumns
        self.Series = self.Series[columns]

    def __apply__zero__timestamp(self, series, initialtimestampmatrix):
        zeroval = series['EXPTIMESTAMP'] - initialtimestampmatrix['INITIALTIMESTAMPBYTRIAL'][series['STIMULUS']]
        return zeroval

    def assemble_index(self):
        self.Series['EXPSAMPLE'] = self.Series.index
        initialvalmatrix = self.Series.groupby('STIMULUS').min()[['EXPTIMESTAMP']]
        initialvalmatrix.columns = ['INITIALTIMESTAMPBYTRIAL']
        self.Series['TRIALTIMESTAMP'] = self.Series.apply(self.__apply__zero__timestamp, axis=1, initialtimestampmatrix=initialvalmatrix)
        maxvalmatrix = self.Series.groupby('STIMULUS').max()[['TRIALTIMESTAMP']]
        maxvalmatrix.columns = ['TRIALDURATION']
        maxvalmatrix['STIMULUS'] = maxvalmatrix.index
        print maxvalmatrix
        self.Series = pandas.merge(self.Series, maxvalmatrix)
        self.Series.set_index(['STIMULUS', 'TRIALTIMESTAMP'], drop=False, inplace=True)

    def zero_timestamp(self):
        zero = self.Series['EXPTIMESTAMP'].iloc[0]
        self.Series['EXPTIMESTAMP'] = self.Series['EXPTIMESTAMP'] - zero

    def flip_axis(self, axis):
        self.Series['R{0}EYE'.format(axis)] = 1 - self.Series['R{0}EYE'.format(axis)]
        self.Series['L{0}EYE'.format(axis)] = 1 - self.Series['L{0}EYE'.format(axis)]

    def filter_negatives(self):
        self.Series['RXEYE'] = self.Series['RXEYE'].mask(self.Series['RXEYE'] < 0, other=None)
        self.Series['LXEYE'] = self.Series['LXEYE'].mask(self.Series['LXEYE'] < 0, other=None)
        self.Series['RYEYE'] = 1 - self.Series['RYEYE'].mask(self.Series['RYEYE'] < 0, other=None)
        self.Series['LYEYE'] = 1 - self.Series['LYEYE'].mask(self.Series['LYEYE'] < 0, other=None)

    def average_eyes(self):
        self.Series['XSERIES'] = self.Series[['RXEYE', 'LXEYE']].mean(axis=1)
        self.Series['YSERIES'] = self.Series[['RYEYE', 'LYEYE']].mean(axis=1)

    def by_stimulus(self, stimulus):
        return self.Series.xs(stimulus, level='STIMULUS')

    def filter_blinks(self):
        print 'Not Implemented'

    def filter_hf(self):
        print 'Not Implemented'

    def recenter_in_epoch(self, source = None, target = None):
        print 'Not Implemented'

    def __row_aoi_inside(self, series, AOICollection = None, rank = None):
        inside = None
        if str(series['STIMULUS']) == str(AOICollection.Name):
            if rank == None:
                inside = AOICollection.is_inside(x=series['XSERIES'], y=series['YSERIES'])
            else:
                inside = False
                for aoi in AOICollection.Contents:
                    if inside == False and int(aoi.Rank) == int(rank) and aoi.is_inside(x=series['XSERIES'], y=series['YSERIES']):
                        inside = True

        return inside

    def __row_time_inside(self, series, window):
        inside = False
        if series[str(window) + '.ONSET'] <= series['TRIALTIMESTAMP'] <= series[str(window) + '.OFFSET']:
            inside = True
        elif series[str(window) + '.ONSET'] <= series['TRIALTIMESTAMP_SHIFT'] <= series[str(window) + '.OFFSET']:
            if stringify(series['STIMULUS'], unfloatify=True) == stringify(series['STIMULUS_SHIFT'], unfloatify=True):
                if math.fabs(series['TRIALTIMESTAMP'] - series[str(window) + '.ONSET']) < math.fabs(series['TRIALTIMESTAMP_SHIFT'] - series[str(window) + '.ONSET']):
                    inside = True
        return inside

    def add_aoi_column(self, aoicollection, byrank = False):
        if byrank == False:
            self.Series[str(aoicollection.Name)] = self.Series.apply(self.__row_aoi_inside, axis=1, AOICollection=aoicollection, rank=None)
        else:
            for rank in aoicollection.Ranks:
                self.Series[str(aoicollection.Name) + '_' + str(rank)] = self.Series.apply(self.__row_aoi_inside, axis=1, AOICollection=aoicollection, rank=rank)

    def add_timing(self, timing):
        mergedframe = pandas.merge(self.Series, timing.Series)
        mergedframe['TRIALTIMESTAMP_SHIFT'] = mergedframe['TRIALTIMESTAMP'].shift(-1)
        mergedframe['STIMULUS_SHIFT'] = mergedframe['STIMULUS'].shift(-1)
        to_drop = ['TRIALTIMESTAMP_SHIFT', 'STIMULUS_SHIFT']
        for window in timing.Windows:
            print 'Processing Epoch {0}'.format(window)
            mergedframe[str(window)] = mergedframe.apply(self.__row_time_inside, axis=1, window=window)

        mergedframe.drop(to_drop, axis=1, inplace=True)
        self.Series = mergedframe

    def write_gaze(self, outfile):
        includedheaders = [ item for item in list(self.Series.columns.values) if item not in self.ExcessColumns ]
        writecsv = self.Series[includedheaders].copy()
        writecsv.to_csv(outfile, index=False)

    def animate_init(self, dot, bar, aoicollection):
        self.Dot = dot
        self.TimeBar = bar
        self.AOICollection = aoicollection
        self.Xseries = list(self.Series['ScreenX'])
        self.Yseries = list(self.Series['ScreenY'])
        self.Dot.center = (-100, -100)
        self.TimeBar.set_width(0)

    def animate_set(self):
        self.Dot.center = (-100, -100)
        self.TimeBar.set_width(0)
        return (self.Dot, self.TimeBar)

    def animate_gaze(self, i):
        self.TimeBar.set_width(self.AvailableWidth * (float(i) / float(self.Frames)))
        if self.Xseries[i] != None and self.Xseries[i] != None:
            self.Dot.center = (self.Xseries[i], self.Yseries[i])
            if self.AOICollection.is_inside(x=self.Xseries[i] / self.AvailableWidth, y=self.Yseries[i] / self.AvailableHeight):
                self.Dot.set_facecolor('green')
            else:
                self.Dot.set_facecolor('red')
        else:
            self.Dot.center = (-100, -100)
        return (self.Dot, self.TimeBar)


class Screen(object):
    """
    The screen class.
    Provides info about the screen settings
    """

    def __init__(self, width = 800, height = 600):
        self.Width = width
        self.Height = height
        self.Area = width * height

    def is_inside(self, x = None, y = None):
        if x == None or y == None:
            return False
        elif x in range(0, self.Width + 1) and y in range(0, self.Height + 1):
            return True
        else:
            return False

    def __str__(self):
        string = 'Screen with parameters:\n    Width: {0}\n    Height: {1}\n    Area: {2}'.format(self.Width, self.Height, self.Area)
        return string

    def plot(self, gaze, AOICollection = None, background = None, timing = None):
        if AOICollection == None:
            AOICollection = AOICollection()
        fig = plt.figure()
        fig.canvas.set_window_title(AOICollection.Name)
        fig.set_dpi(100)
        fig.set_size_inches(self.Width / 100, self.Height / 100)
        ax = plt.axes(xlim=(0, self.Width), ylim=(0, self.Height))
        if background != None and exists(background):
            if os.path.isfile(cleanPathString(background)):
                rawimage = np.array(Image.open(cleanPathString(background)))
                if rawimage.size == 0:
                    print 'Image loaded is empty'
                    sys.exit(1)
                print 'Loading Image {0}'.format(background)
                image = Image.open(cleanPathString(background))
                image = image.resize((self.Width, self.Height), Image.ANTIALIAS).transpose(Image.FLIP_TOP_BOTTOM)
                ax.imshow(image, interpolation='bicubic', origin='lower')
        else:
            ax.set_axis_bgcolor('#000000')
        gazedot = ptc.Circle((self.Width / 2, self.Height / 2), 10, fc='red', alpha=0.75)
        timebar = ptc.Rectangle((0, 0), 0, self.Height * 0.02, alpha=0.5, ec='none', fc='green')
        ax.add_patch(gazedot)
        ax.add_patch(timebar)
        if AOICollection != None:
            print AOICollection
            for aoi in AOICollection.Contents:
                if aoi.Type == 'Rectangle':
                    print 'Adding a rectanglular AOI to the image.'
                    ax.add_artist(ptc.Rectangle((aoi.Xmin * self.Width, aoi.Ymin * self.Height), (aoi.Xmax - aoi.Xmin) * self.Width, (aoi.Ymax - aoi.Ymin) * self.Height, alpha=0.5, color='#ffffff'))
                    rank = ax.text(aoi.Xcenter * self.Width, aoi.Ycenter * self.Height, str(aoi.Rank), ha='center', va='center', size=self.Width / 10, alpha=0.5)
                elif aoi.Type == 'Ellipse':
                    print 'Adding an elliptical AOI to the image.'
                    ax.add_artist(ptc.Ellipse((aoi.Xcenter * self.Width, aoi.Ycenter * self.Height), aoi.Xradius * 2 * self.Width, aoi.Yradius * 2 * self.Height, alpha=0.5, color='#ffffff'))
                    rank = ax.text(aoi.Xcenter * self.Width, aoi.Ycenter * self.Height, str(aoi.Rank), ha='center', va='center', size=self.Width / 10, alpha=0.5)

        if gaze.Frames > 0:
            print 'Adding the gaze to the image.'
            gaze.animate_init(gazedot, timebar, AOICollection)
            gaze_animation = animation.FuncAnimation(fig, gaze.animate_gaze, init_func=gaze.animate_set, frames=gaze.Frames, interval=1 / gaze.Hertz, blit=True)
        else:
            print 'Quality gaze data not available for this image.'
            gazedot.center = (-100, -100)
        plt.show()


class AOICollection(object):
    """
    The AOICollection class.
    Defines a collection of AOIs, likely all relating to a given stimulus.
    """

    def __init__(self, contents = [], name = 'Group', obt = None, json = None, screen = Screen()):
        self.Ranks = set([])
        self.Contents = contents
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

    def write_obt(self, path, screen):
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

    def write_json(self, path):
        try:
            contents = []
            for AOI in self.Contents:
                contents.append(AOI.__dict__())

            with open(path, 'w') as outfile:
                json.dump(contents, outfile, sort_keys=True, indent=4, ensure_ascii=False)
            return True
        except:
            raise IOError('Could not write file to {0}'.format(path))

    def write_image(self, path, screen):
        fig = plt.figure(frameon=False)
        fig.set_dpi(100)
        fig.set_size_inches(screen.Width / 100, screen.Height / 100)
        ax = fig.add_axes([0,
         0,
         1,
         1])
        ax.axis('off')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.add_artist(ptc.Rectangle((0, 0), 1, 1, alpha=1, color='#000000'))
        for aoi in self.Contents:
            if aoi.Type == 'Rectangle':
                ax.add_artist(ptc.Rectangle((aoi.Xmin, aoi.Ymin), aoi.Xmax - aoi.Xmin, aoi.Ymax - aoi.Ymin, alpha=1, color='#ffffff'))
            elif aoi.Type == 'Ellipse':
                ax.add_artist(ptc.Ellipse((aoi.Xcenter, aoi.Ycenter), aoi.Xradius * 2, aoi.Yradius * 2, alpha=1, color='#ffffff'))

        try:
            fig.savefig(path, bbox_inches=None, origin='lower', extent=[0,
             1,
             0,
             1])
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
