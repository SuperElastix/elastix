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
    parser.add_option( "-b", "--baseline", dest="baseline", help="baseline checksum" )

    (options, args) = parser.parse_args()

    # Equivalent to: fileName = options.directory + "/" + "elastix.log"
    fileName = os.path.join( options.directory, "elastix.log" );

    # Read elastix.log and find last line with checksum
    f = open( fileName )    
    for line in f:
      if "Registration result checksum:" in line:
        checksumline = line

    # Extract checksum
    checksum = checksumline.split(': ')[1].rstrip( "\n" );

    # Print result
    print "The registration result checksum is: %s" % checksum
    print "The baseline values is: %s" % options.baseline

    if options.baseline != checksum:
      print "FAILURE: These values are NOT the same.\n"
      if options.verbose:
        print "Complete elastix.log file:\n"
        f.seek(0,0)
        for line in f:
          print "%s" % line,
      f.close();
      return 1

    f.close();
    print "SUCCESS: These values are the same."
    return 0

#-------------------------------------------------------------------------------
if __name__ == '__main__':
    sys.exit(main())
