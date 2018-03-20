import sys
import os
import os.path
import glob
from optparse import OptionParser

#-------------------------------------------------------------------------------
# the main function
# cd bin_VS2010
# ctest -C Release
# cd Testing
# python ../../elastix/Testing/elx_get_checksum_list.py -l elastix_run*
# cd ..
# cmake .
# ctest -C Release -R COMPARE_CHECKSUM
# svn commit -m "ENH: updating baselines after recent change X"
def main():
    # usage, parse parameters
    usage = "usage: %prog [options] arg"
    parser = OptionParser( usage )

    # option to debug and verbose
    parser.add_option( "-v", "--verbose",
      action="store_true", dest="verbose" )

    # options to control files
    parser.add_option( "-l", "--list", type="string", dest="directoryList", help="list of elastix output directories" )

    (options, args) = parser.parse_args()

    # Check if option -l is given
    if options.directoryList == None :
      parser.error( "The option directory list (-l) should be given" )

    # Use glob, this works not only on Linux
    dirList = glob.glob( options.directoryList );
    # Add everything not processed
    dirList.extend( args );

    print( "directory checksum" )
    for directory in dirList:
      # Equivalent to: fileName = options.directory + "/" + "elastix.log"
      fileName = os.path.join( directory, "elastix.log" );

      # Read elastix.log and find last line with checksum
      try:
        f = open( fileName )
      except IOError as e:
        print( directory + " No elastix.log found" )
        continue

      checksumFound = False;
      for line in f:
        if "Registration result checksum:" in line:
          checksumline = line;
          checksumFound = True;

      # Extract checksum
      if checksumFound:
        checksum = checksumline.split(': ')[1].rstrip( "\n" );

        # Print result
        print( directory + " " + checksum );
      else:
        print( directory + " -" );

      f.close();

    return 0

#-------------------------------------------------------------------------------
if __name__ == '__main__':
    sys.exit(main())
