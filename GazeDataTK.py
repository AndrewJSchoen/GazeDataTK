
#============================================================================
#============ General Utility ===============================================

Version = "1.0.0"
import sys, os, numpy, pandas, math
from docopt import docopt
from AOI import *

#============================================================================
#============ General Utility ===============================================

doc = """
GazeData ToolKit.

Usage:
  GazeDataTK.py <command> [options] [--transfercolumn=column...] [--addtiming --timingfile=file]

Commands:
  View
  Process
  Convert

Options:
  -h --help                    Show this screen.
  -v --version                 Show the current version.
  --stimulusid=column          The column name in the input file referencing the stimulus' ID column of the reference file [default: image]
  --rxeye=column               The column name in the input file of the right eye's x coordinates. [default: XGazePosRightEye]
  --lxeye=column               The column name in the input file of the left eye's x coordinates. [default: XGazePosLeftEye]
  --ryeye=column               The column name in the input file of the right eye's y coordinates. [default: YGazePosRightEye]
  --lyeye=column               The column name in the input file of the left eye's y coordinates. [default: YGazePosLeftEye]
  --rvalidity=column           The column name in the input file of the right eye's validity. [default: ValidityRightEye]
  --lvalidity=column           The column name in the input file of the left eye's validity. [default: ValidityLeftEye]
  --rpupildiameter=column      The column name in the input file of the right eye's pupil diameter. [default: DiameterPupilRightEye]
  --lpupildiameter=column      The column name in the input file of the left eye's pupil diameter. [default: DiameterPupilLeftEye]
  --transfercolumn=column      A column to move from the original gazedata file to the output.
  --timingcolumn=column        A column contining an absolute timestamp for each collection event in the gaze data. [default: TETTime]
  --screenwidth=width          The width of the screen, [default: 800]
  --screenheight=height        The height of the screen, [default: 600]
  --hertz=hertz                Hertz of data collection, [default: 120]
  --yflip=True/False           True if gaze data values increase as gaze moves down the image. [default: True]
  --xflip=True/False           True if gaze data values increase as gaze moves left on the image. [default: False]
  --reference=referencefile    Manifest of files designating AOIs, images.
  --intype=type                Input Type of AOI collection file (OBT or JSON).  Will attempt to determine from infile.
  --infile=file                Path to input file for AOI collection (Convert) or GazeData file (View, Process).
  --timingfile=file            Path to timing file (View, Process), contains a row for each stimulus, and two columns for each epoch (e.g. Fixation.Onset and Fixation.Offset), or any other columns associated with all elements for a stimulus, (e.g. TrialType).
  --timingunits=units          Units of time for timingfile, options are seconds, milliseconds, samples; [default: milliseconds]
  --outtype=type               Output Type of AOI collection file (OBT, JSON, PNG, PDF, or JPG). Will attempt to determine from outfile.
  --outfile=file               Path to output file for AOI collection (Convert command) or extended GazeData file (Process command).
  --collectionname=string      Name of the collection. By default the same as the input file.
  --aoibyrank                  Add columns for each AOI for each rank. Columns will be named Name_Rank. [default: False]
  --addtiming                  Add a column for each epoch, containing time spent in that stimulus epoch, as indicated in the timingfile. [default: False]
  --recenterby=epoch           Epoch (e.g. Fixation) to use in determing center for the rest of the stimulus. [default: Fixation]

Input (GazeData) File:
  Standard GazeData file from Tobii/E-Prime. Contains a column corresponding in name to the value specified with --stimulusid.

Reference File CSV:
  CSV file with a column "ID", "File", and "Image"
"""

def exists(path):
  #Shorthand for checking the existence of a file
  if os.path.exists(cleanPathString(path)):
      return(1)
  else:
      return(0)

def RepresentsInt(s):
	try:
		int(s)
		return True
	except ValueError:
		return False

def removeDuplicates(inputlist):
  output = []
  for x in inputlist:
    if x not in output:
      output.append(x)
  return output

def cleanPathString(path):
  if path.endswith("\n"):
    path = path[:-1]
  if path.endswith("/"):
    path = path[:-1]
  if path.startswith("="):
    path = path[1:]
  if path.startswith("~"):
    path = os.path.expanduser(path)
  realpath = os.path.realpath(path)
  return realpath




def clean(rawarguments):
    cleanedarguments = {}
    cleanedarguments["COMMAND"] = rawarguments ["<command>"]
    if cleanedarguments["COMMAND"] not in ["View", "Process", "Convert"]:
        print("{0} is not a valid command. Exiting".format(cleanedarguments["COMMAND"]))
        sys.exit(1)
    cleanedarguments["STIMULUS"] = rawarguments["--stimulusid"]
    if RepresentsInt(rawarguments["--screenwidth"]) and RepresentsInt(rawarguments["--screenheight"]):
        cleanedarguments["SCREEN"] = Screen(int(rawarguments["--screenwidth"]),int(rawarguments["--screenheight"]))
    else:
        print("The screen width or height you specified is not a valid integer value.")
        sys.exit(1)

    if cleanedarguments["COMMAND"] == "View":
        ioset = ["--infile"]
    else:
        ioset = ["--infile", "--outfile"]

    #IN/OUT
    for io in ioset:
        if rawarguments[io] == None:
            print("You did not supply all of the needed arguments for {0}.".format(cleanedarguments["COMMAND"]))
            sys.exit(1)
    cleanedarguments["INFILE"] = cleanPathString(rawarguments["--infile"])
    if cleanedarguments["COMMAND"] != "View":
        cleanedarguments["OUTFILE"] = cleanPathString(rawarguments["--outfile"])


    if cleanedarguments["COMMAND"] in ["View", "Process"]:
        cleanedarguments["RXEYE"] = rawarguments["--rxeye"]
        cleanedarguments["LXEYE"] = rawarguments["--lxeye"]
        cleanedarguments["RYEYE"] = rawarguments["--ryeye"]
        cleanedarguments["LYEYE"] = rawarguments["--lyeye"]
        cleanedarguments["RVALIDITY"] = rawarguments["--rvalidity"]
        cleanedarguments["LVALIDITY"] = rawarguments["--lvalidity"]
        cleanedarguments["RPUPILDIAMETER"] = rawarguments["--rpupildiameter"]
        cleanedarguments["LPUPILDIAMETER"] = rawarguments["--lpupildiameter"]
        cleanedarguments["YFLIP"] = rawarguments["--yflip"]
        cleanedarguments["XFLIP"] = rawarguments["--xflip"]
        cleanedarguments["HERTZ"] = int(rawarguments["--hertz"])
        cleanedarguments["TIMESTAMP"] = rawarguments["--timingcolumn"]
        cleanedarguments["TRANSFERCOLUMNS"] = rawarguments["--transfercolumn"]
        cleanedarguments["ADDTIMING"] = rawarguments["--addtiming"]
        if cleanedarguments["ADDTIMING"]:
            cleanedarguments["TIMINGFILE"] = cleanPathString(rawarguments["--timingfile"])
        if rawarguments["--timingunits"].lower() in ["seconds", "milliseconds"]:
            cleanedarguments["TIMINGUNITS"] = rawarguments["--timingunits"].lower()
        else:
            print("{0} is not a recognized timing unit. Choose either 'seconds' or 'milliseconds'. Defaulting to milliseconds.")
            cleanedarguments["TIMINGUNITS"] = "milliseconds"


        dfdict = {}
        headerset = []
        for column in cleanedarguments["TRANSFERCOLUMNS"]:
            headerset.append({"rawcolumn": column, "newcolumn": column})
        for column in ["RXEYE", "LXEYE", "RYEYE", "LYEYE", "RVALIDITY", "LVALIDITY", "RPUPILDIAMETER", "LPUPILDIAMETER", "TIMESTAMP", "STIMULUS"]:
            headerset.append({"rawcolumn": cleanedarguments[column], "newcolumn": column})

        if exists(cleanedarguments["INFILE"]) and exists(rawarguments["--reference"]):
            rawgazedata = pandas.read_table(cleanedarguments["INFILE"])
            for column in headerset:
                if column["rawcolumn"] in list(rawgazedata.columns.values):
                    dfdict[column["newcolumn"]] = rawgazedata[column["rawcolumn"]]
                else:
                    print("[WARNING] Column '{0}' not present in originial gaze data file. Not transferring.".format(column["rawcolumn"]))
            if not set(["RXEYE", "LXEYE", "RYEYE", "LYEYE"]).issubset(set(dfdict.keys())):
                print("The gazedata did not contain critical columns, or you did not specify the columns correctly.")
                sys.exit(1)
            cleanedarguments["GAZEFRAME"] = pandas.DataFrame(dfdict)
            #floatify the timestamp column
            cleanedarguments["GAZEFRAME"]["TIMESTAMP"] = cleanedarguments["GAZEFRAME"]["TIMESTAMP"].apply(floatify)
            if cleanedarguments["TIMINGUNITS"] == "seconds":
                cleanedarguments["GAZEFRAME"]["TIMESTAMP"] = cleanedarguments["GAZEFRAME"]["TIMESTAMP"]*1000
            cleanedarguments["MANIFEST"] = pandas.read_csv(cleanPathString(rawarguments["--reference"]), index_col="ID")
        else:
            print("The gazedata or reference file did not exist at the location specified.")
            sys.exit(1)

        if cleanedarguments["ADDTIMING"]:
            if exists(cleanedarguments["TIMINGFILE"]):
                rawtimingdata = pandas.read_csv(cleanedarguments["TIMINGFILE"])
                cleanedarguments["TIMINGDATA"] = Timing(rawtimingdata, cleanedarguments["TIMINGUNITS"], cleanedarguments["STIMULUS"])
            else:
                print("[ERROR] Timing file not found! Exiting!")
                sys.exit(1)


    if cleanedarguments["COMMAND"] == "Process":
        if rawarguments["--aoibyrank"] == False or rawarguments["--aoibyrank"] == "False":
            cleanedarguments["AOIBYRANK"] = False
        else:
            cleanedarguments["AOIBYRANK"] = True

    if cleanedarguments["COMMAND"] == "Convert":


        #Handle Extensions
        cleanedarguments["INTYPE"] = None
        cleanedarguments["OUTTYPE"] = None

        inextension = os.path.splitext(cleanedarguments["INFILE"])[1][1:].upper()
        if rawarguments["--intype"] != None:
            if rawarguments["--intype"].upper() in ["OBT", "JSON"]:
                cleanedarguments["INTYPE"] = rawarguments["--intype"].upper()
            elif inextension in ["OBT", "JSON"]:
                cleanedarguments["INTYPE"] = inextension
        elif rawarguments["--intype"] == None and inextension in ["OBT", "JSON"]:
            cleanedarguments["INTYPE"] = inextension

        outextension = os.path.splitext(cleanedarguments["OUTFILE"])[1][1:].upper()
        if rawarguments["--outtype"] != None:
            if rawarguments["--outtype"].upper() in ["OBT", "JSON", "PNG", "JPG", "PDF"]:
                cleanedarguments["OUTTYPE"] = rawarguments["--outtype"].upper()
            elif outextension in ["OBT", "JSON"]:
                cleanedarguments["OUTTYPE"] = outextension
        elif rawarguments["--outtype"] == None and outextension in ["OBT", "JSON", "PNG", "JPG", "PDF"]:
            cleanedarguments["OUTTYPE"] = outextension

        if cleanedarguments["INTYPE"] == None or cleanedarguments["OUTTYPE"] == None:
            print("You did not provide a valid file type for input or output.")
            sys.exit(1)

        if rawarguments["--collectionname"] == None:
            cleanedarguments["COLLECTION"] = os.path.splitext(os.path.basename(cleanedarguments["INFILE"]))[0]
        else:
            cleanedarguments["COLLECTION"] = rawarguments["--collectionname"]
    return cleanedarguments

def process(arguments):
    #Get images list
    images = removeDuplicates(list(arguments["GAZEFRAME"]["STIMULUS"]))
    for image in images:
        if image not in list(arguments["MANIFEST"].index.values):
            print("Warning: Image {0} does not have a corresponding AOI in the AOI manifest file.".format(image))

    #Add Timing values, if necessary
    if arguments["ADDTIMING"]:
        timingval = arguments["TIMINGDATA"]
        #print(arguments["TIMINGDATA"])
    else:
        timingval = None

    #Get a gaze object
    gaze = Gaze(arguments["GAZEFRAME"], arguments["SCREEN"], timing=timingval, hertz=arguments["HERTZ"], yflip=arguments["YFLIP"], xflip=arguments["XFLIP"])

    #Loop over AOIs
    for image in sorted(images):
        print("Processing AOI {0}".format(image))
        aoisfilepath = arguments["MANIFEST"].ix[image]["File"]
        if exists(aoisfilepath):
            extension = os.path.splitext(aoisfilepath)[1]
            if extension == ".OBT":
                obtval = aoisfilepath
                jsonval = None
            elif extension == ".json":
                obtval = None
                jsonval = aoisfilepath
            else:
                print("File {0} found, but it is not one of the expected file types (OBT or json)".format(aoisfilepath))
                continue
            with AOICollection(name=str(image), obt=obtval, json=jsonval, screen=arguments["SCREEN"]) as aoicollection:
                gaze.add_aoi_column(aoicollection, byrank=arguments["AOIBYRANK"])
        else:
            print("[WARNING] File {0} not found!".format(aoisfilepath))
    gaze.write_gaze(arguments["OUTFILE"])

def convert(arguments):
    #Load input file
    print("Loading {0} file.".format(arguments["INTYPE"]))
    if arguments["INTYPE"] == "JSON":
        aoicollection =  AOICollection(name=str(arguments["COLLECTION"]), json=arguments["INFILE"])
    elif arguments["INTYPE"] == "OBT":
        aoicollection =  AOICollection(name=str(arguments["COLLECTION"]), obt=arguments["INFILE"], screen=arguments["SCREEN"])
    else:
        print("The input file type specified is not recognized. Type should be either 'OBT' or 'JSON'.")
        sys.exit(1)
    print(aoicollection)
    print("Writing {0} file.".format(arguments["OUTTYPE"]))
    if arguments["OUTTYPE"] == "JSON":
        aoicollection.write_json(arguments["OUTFILE"])
    elif arguments["OUTTYPE"] == "OBT":
        aoicollection.write_obt(arguments["OUTFILE"], screen=arguments["SCREEN"])
    elif arguments["OUTTYPE"] in ["PNG", "JPG", "PDF"]:
        aoicollection.write_image(arguments["OUTFILE"], screen=arguments["SCREEN"])
    else:
        print("The output file type specified is not recognized. Type should be either 'OBT' or 'JSON'.")
        sys.exit(1)

def formatGaze(rawgaze, hertz, AOICollection=None, yflip=True, xflip=False):
    if AOICollection != None:
        imagedataslice = rawgaze[rawgaze["STIMULUS"].isin([int(AOICollection.Name)])]
    else:
        imagedataslice = rawgaze
    gaze = Gaze(imagedataslice, arguments["SCREEN"], hertz=hertz, yflip=yflip, xflip=xflip)
    return gaze

def view(arguments):
    images = removeDuplicates(list(arguments["GAZEFRAME"]["STIMULUS"]))

    for image in images:
        if image not in list(arguments["MANIFEST"].index.values):
            print("Warning: Image {0} does not have a corresponding AOI in the AOI manifest file.".format(image))
    for image in images:
        if image not in list(arguments["MANIFEST"].index.values):
            print("That image is not present in the AOI manifest file. No AOIs will be loaded.")
            background = None
        else:
            background = arguments["MANIFEST"].ix[image]["Image"]

        aoisfilepath = arguments["MANIFEST"].ix[image]["File"]
        if exists(aoisfilepath):
            extension = os.path.splitext(aoisfilepath)[1]
            if extension == ".OBT":
                obtval = aoisfilepath
                jsonval = None
            elif extension == ".json":
                obtval = None
                jsonval = aoisfilepath
            else:
                print("File {0} found, but it is not one of the expected file types (OBT or json)".format(aoisfilepath))
                jsonval = None
                obtval = None
        else:
            obtval = None
            jsonval = None

        with AOICollection(name=str(image), obt=obtval, json=jsonval, screen=arguments["SCREEN"]) as aoicollection:
            gaze = formatGaze(arguments["GAZEFRAME"], arguments["HERTZ"], AOICollection=aoicollection, yflip=arguments["YFLIP"], xflip=arguments["XFLIP"])
            arguments["SCREEN"].plot(gaze, AOICollection=aoicollection, background=background)


if __name__ == '__main__':
    args = sys.argv
    del args[0]
    arguments = clean(docopt(doc, argv=args, version='GazeData ToolKit v{0}'.format(Version)))
    if arguments["COMMAND"] == "View":
        view(arguments)
    elif arguments["COMMAND"] == "Convert":
        convert(arguments)
    else:
        process(arguments)
