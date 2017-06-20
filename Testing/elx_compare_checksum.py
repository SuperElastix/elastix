import sys
import os
import os.path
from optparse import OptionParser

#-------------------------------------------------------------------------------
# the main function
# This python script compares are registration result with a baseline value.
# The registration result is summarized by the registration checksum as printed
# in the elastix.log file. The baseline checksum value is given through the
# command line, either via -b checksum or via -f filename containing the baseline
# value. The latter file can be automatically generated per system via the python
# script elx_get_checksum_list.py.
def main() :
    # usage, parse parameters
    usage = "usage: %prog [options] arg";
    parser = OptionParser( usage );

    # option to debug and verbose
    parser.add_option( "-v", "--verbose",
        action="store_true", dest="verbose" );

    # options to control files
    parser.add_option( "-d", "--directory", dest="directory", help="elastix output directory" );
    parser.add_option( "-b", "--baseline", dest="baseline", help="baseline checksum" );
    parser.add_option( "-f", "--baselinefile", dest="baselineFile", help="baseline file" );

    (options, args) = parser.parse_args();

    # Check if option -d is given
    if options.directory == None :
      parser.error( "The option directory (-d) should be given" );

    # Check if either option -b or -f is given
    if options.baseline != None and options.baselineFile != None :
      parser.error( "Use either option -d OR -f" );
    if options.baseline == None and options.baselineFile == None :
      parser.error( "Use either option -d OR -f" );

    # Get the baseline checksum. Either it is provided through the commandline
    # by -b, or by means of a text file using -f.
    if options.baselineFile != None :
      if os.path.exists( options.baselineFile ) :
        fb = open( options.baselineFile );
        checksumlineBaseline = "";
        dirString = options.directory.rsplit( "/", 1 )[1];
        for line in fb :
          if dirString == line.split()[0].rsplit("/")[-1] :
            checksumlineBaseline = line;
        fb.close();

        # Extract checksum
        tmp = checksumlineBaseline.split(' ');
        if len( tmp ) > 1 :
          baselineChecksum = tmp[1].rstrip( "\r\n" );
        else :
          print( "ERROR: the baseline checksum is not found!" );
          return 1;
      else :
        print( "ERROR: the checksum baseline file " + options.baselineFile + " does not exist!" );
        return 1;
    else :
      baselineChecksum = options.baseline;

    # todo check if baselineChecksum now is defined to something

    # Equivalent to: fileName = options.directory + "/" + "elastix.log"
    elastixLogFileName = os.path.join( options.directory, "elastix.log" );

    # Read elastix.log and find last line with checksum
    f = open( elastixLogFileName );
    checksumline = "";
    for line in f :
      if "Registration result checksum:" in line :
        checksumline = line;

    # Extract checksum
    tmp = checksumline.split(': ');
    if len( tmp ) > 1 :
      testChecksum = tmp[1].rstrip( "\r\n" );
    else :
      testChecksum = "not found";

    # Print result
    print( "The registration result checksum is: " + testChecksum );
    print( "The baseline checksum is:            " + baselineChecksum );

    if baselineChecksum != testChecksum :
      print( "FAILURE: These values are NOT the same.\n" );
      if options.verbose:
        print( "Complete elastix.log file:\n" );
        f.seek(0,0);
        for line in f :
          if "(TransformParameters" not in line : print( line );
          else : print( "(TransformParameters <values have been removed to avoid CDash trunctation> )" );
      f.close();
      return 1;

    f.close();
    print( "SUCCESS: These values are the same." );
    return 0

#-------------------------------------------------------------------------------
if __name__ == '__main__':
    sys.exit(main())
