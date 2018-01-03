import sys, subprocess
import os
import os.path
import re
import shutil
import math
#import numpy
from optparse import OptionParser

#-------------------------------------------------------------------------------
# the main function
def main():
  # usage, parse parameters
  usage = "usage: %prog [options] arg";
  parser = OptionParser( usage );

  # option to debug and verbose
  parser.add_option( "-v", "--verbose", action="store_true", dest="verbose" );

  # options to control files
  parser.add_option( "-d", "--directory", dest="directory", help="elastix output directory" );
  parser.add_option( "-f", "--fixedlandmarks", dest="flm", help="fixed image landmarks" );
  parser.add_option( "-b", "--baselinetp", dest="btp", help="baseline transform parameter file" );
  parser.add_option( "-t", "--threshold", dest="threshold", help="threshold on landmark error" );
  parser.add_option( "-p", "--path", dest="path", help="path where executables can be found" );

  (options, args) = parser.parse_args();

  # Check if option -d and -f and -b are given
  if options.directory == None :
    parser.error( "The option directory (-d) should be given" );
  if options.flm == None :
    parser.error( "The option directory (-f) should be given" );
  if options.btp == None :
    parser.error( "The option directory (-b) should be given" );
  if options.threshold == None : threshold = 0.5;
  else :                         threshold = options.threshold;

  # Get the transform parameters files
  tpFileName   = os.path.join( options.directory, "TransformParameters.0.txt" );
  tpFileName_b = options.btp;

  # Sanity checks
  if not os.path.exists( tpFileName ) :
    print( "ERROR: the file " + tpFileName + " does not exist" );
    return 1;

  # Below we use programs that are compiled with elastix, and are thus available
  # in the binary directory. The user of this script has to supply the path
  # to the binary directory via the command line.
  # In order to make sure that python is able to find these programs we add
  # the paths to the local environment.
  _path = os.path.dirname( options.path );
  _path += os.pathsep + os.getenv('PATH');
  os.environ['PATH'] = _path;

  # output file of the transformix runs; and copies, for later debugging.
  landmarkstemp = os.path.join( options.directory, "outputpoints.txt" );
  landmarks1full = os.path.join( options.directory, "outputpoints_current.txt" );
  landmarks2full = os.path.join( options.directory, "outputpoints_baseline.txt" );

  # Remove copies otherwise os.rename will not work on Windows:
  # "On Windows, if dst already exists, OSError will be raised"
  if( os.path.exists( landmarks1full ) ) : os.remove( landmarks1full );
  if( os.path.exists( landmarks2full ) ) : os.remove( landmarks2full );

  #
  # Transform the fixed image landmarks by the current result
  #
  print( "Transforming fixed image landmarks using " + tpFileName );
  landmarks1 = os.path.join( options.directory, "landmarks_current.txt" );
  subprocess.call( [ "transformix", "-def", options.flm, "-out", options.directory, "-tp", tpFileName ],
    stdout=subprocess.PIPE );

  # Parse file to extract only the column with the output points
  f1 = open( landmarkstemp, 'r' );
  f2 = open( landmarks1, 'w' );
  for line in f1 :
    f2.write( line.strip().split(';')[4].strip().strip( "OutputPoint = [ " ).rstrip( " ]" ) + "\n" );
  f1.close(); f2.close();
  shutil.move( landmarkstemp, landmarks1full ); # for later inspection

  #
  # Transform the fixed image landmarks by the baseline result
  #
  print( "Transforming fixed image landmarks using " + tpFileName_b );
  landmarks2 = os.path.join( options.directory, "landmarks_baseline.txt" );
  subprocess.call( [ "transformix", "-def", options.flm, "-out", options.directory, "-tp", tpFileName_b ],
    stdout=subprocess.PIPE );
  # shutil.copyfile( landmarkstemp, landmarks2 ); // this should not be necessary

  # Parse file to extract only the column with the output points
  f1 = open( landmarkstemp, 'r' );
  f2 = open( landmarks2, 'w' );
  for line in f1 :
    f2.write( line.strip().split(';')[4].strip().strip( "OutputPoint = [ " ).rstrip( " ]" ) + "\n" );
  f1.close(); f2.close();
  shutil.move( landmarkstemp, landmarks2full ); # for later inspection

  # Compute the distance between all transformed landmarks
  f1 = open( landmarks1, 'r' ); f2 = open( landmarks2, 'r' );
  distances = [];
  for line1, line2 in zip( f1, f2 ) :
    floats1 = [ float(x) for x in line1.split() ];
    floats2 = [ float(x) for x in line2.split() ];
    diffSquared = [ (m - n) * (m - n) for m, n in zip( floats1, floats2 ) ];
    distance = math.sqrt( sum( diffSquared ) );
    distances.append( distance );

  # Compute some statistics on the distances
  distances.sort();
  minDistance  = "{0:.3f}".format( distances[ 0 ] );
  Q1           = "{0:.3f}".format( distances[ int( len( distances ) * 1.0 / 4.0 ) ] );
  medDistance  = "{0:.3f}".format( distances[ int( len( distances ) * 2.0 / 4.0 ) ] );
  Q3           = "{0:.3f}".format( distances[ int( len( distances ) * 3.0 / 4.0 ) ] );
  maxDistance  = "{0:.3f}".format( distances[ -1 ] );
  meanDistance = "{0:.3f}".format( sum( distances ) / float( len( distances ) ) );

  # With numpy it would be:
  #l1 = numpy.loadtxt( landmarks1 );
  #l2 = numpy.loadtxt( landmarks2 );
  #meandistance = numpy.mean( numpy.sum( (l1-l2)**2, axis=-1)**0.5 );

  # Report
  print( "The landmark distance between current and baseline is:" );
  print( "min   | Q1    | med   | Q3    | max   | mean" );
  print( minDistance + " | " +  Q1 + " | " +  medDistance + " | " +  Q3 + " | " + maxDistance + " | " +  meanDistance  );
  if float( Q3 ) < float( threshold ) :
    print( "SUCCESS: third quartile landmark distance is lower than " + str( threshold ) + " mm" );
    return 0;
  else :
    print( "FAILURE: third quartile landmark distance is higher than " + str( threshold ) + " mm" );
    return 1;

#-------------------------------------------------------------------------------
if __name__ == '__main__':
    sys.exit(main())
