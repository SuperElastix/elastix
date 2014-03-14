import sys
import os
import os.path
from optparse import OptionParser

#-------------------------------------------------------------------------------
# the main function
def main():
    # usage, parse parameters
    usage = "usage: %prog [options] arg"
    parser = OptionParser( usage )

    # option to debug and verbose
    parser.add_option( "-v", "--verbose",
        action="store_true", dest="verbose" )

    # options to control files
    parser.add_option( "-d", "--directory", dest="directory", help="elastix output directory" )
    parser.add_option( "-b", "--baseline", dest="baseline", help="baseline string" )

    (options, args) = parser.parse_args()

    # loop in directory and find latest IterationInfo
    largestElastixLevel = -1
    largestResolutionLevel = -1
    latestFile = ""
    for subdir in os.walk( options.directory ):
        for i in subdir[2]:
            index = i.find( "IterationInfo" )
            if index != -1:
                fileNameParts = i.split( "." )
                currentElastixLevel = int( fileNameParts[1] )
                currentResolutionLevel = int( fileNameParts[2].lstrip('R') )

                if currentElastixLevel > largestElastixLevel:
                    largestElastixLevel = currentElastixLevel
                    largestResolutionLevel = currentResolutionLevel
                    latestFile = i
                elif currentElastixLevel == largestElastixLevel and currentResolutionLevel > largestResolutionLevel:
                    largestResolutionLevel = currentResolutionLevel
                    latestFile = i

    # Sanity check
    if latestFile == "":
      print( "ERROR: no IterationInfo files found in '" + options.directory + "'" )
      return 1

    print( "The latest iteration file is '" + latestFile + "'" )

    # Read last line of IterationInfo file
    fileName = options.directory + "/" + latestFile
    f = open( fileName )
    lineList = f.readlines()
    f.close()
    lastLine = lineList[ len(lineList) - 1 ]

    # Split the last line and the baseline
    lastLine = lastLine.rstrip( "\n" )
    print( "The final registration result has values:\n'" + lastLine + "'" )
    print( "The baseline values are:\n'" + options.baseline + "'" )

    lastLineValues = lastLine.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
    lastLineValues = lastLineValues.split(' ')

    baselineValues = options.baseline.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
    baselineValues = baselineValues.split(' ')


    index = 0
    for i in baselineValues:
        if i == 'x':
            index = index + 1
            continue

        fValueBaseline = float( i )
        fValueLastLine = float( lastLineValues[ index ] )

        if fValueBaseline != fValueLastLine:
            print( "ERROR: These lines are NOT the same" )
            return 1;

        index = index + 1

    print( "SUCCESS: These lines are the same" )
    return 0

#-------------------------------------------------------------------------------
if __name__ == '__main__':
    sys.exit(main())
