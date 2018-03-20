import sys, subprocess
import os
import os.path
import shutil
import re
import glob
from optparse import OptionParser

#-------------------------------------------------------------------------------
# the main function
# Below we deform the moving image segmentation by the current result as well as
# by a previous stored result. This makes this test a regression test.
#
# We could instead compare with a fixed image segmentation, but that would require
# the tested registrations to be relatively good, which they are not to save time.

def main():
  # usage, parse parameters
  usage = "usage: %prog [options] arg";
  parser = OptionParser( usage );

  # option to debug and verbose
  parser.add_option( "-v", "--verbose", action="store_true", dest="verbose" );

  # options to control files
  parser.add_option( "-d", "--directory", dest="directory", help="elastix output directory" );
  parser.add_option( "-m", "--movingsegmentation", dest="mseg", help="moving image segmentation" );
  parser.add_option( "-b", "--baselinetp", dest="btp", help="baseline transform parameter file" );
  parser.add_option( "-p", "--path", dest="path", help="path where executables can be found" );

  (options, args) = parser.parse_args();

  # Check if option -d and -m and -b are given
  if options.directory == None :
    parser.error( "The option directory (-d) should be given" );
  if options.mseg == None :
    parser.error( "The option directory (-m) should be given" );
  if options.btp == None :
    parser.error( "The option directory (-b) should be given" );

  # Get the transform parameters files
  tpFileName_in   = os.path.join( options.directory, "TransformParameters.0.txt" );
  tpFileName      = os.path.join( options.directory, "TransformParameters.seg.txt" );
  tpFileName_b_in = options.btp;
  tpFileName_b    = os.path.join( options.directory, "TransformParameters.baseline.seg.txt" );

  # Sanity checks
  if not os.path.exists( tpFileName_in ) :
    print( "ERROR: the file " + tpFileName_in + " does not exist" );
    return 1;

  # Below we use programs that are compiled with elastix, and are thus available
  # in the binary directory. The user of this script has to supply the path
  # to the binary directory via the command line.
  # In order to make sure that python is able to find these programs we add
  # the paths to the local environment.
  _path = os.path.dirname( options.path );
  _path += os.pathsep + os.getenv('PATH');
  os.environ['PATH'] = _path;

  #
  # Deform the moving image segmentation by the current result
  #
  print( "Deforming moving image segmentation using " + tpFileName_in );

  # Make the transform parameters file suitable for binary images
  f1 = open( tpFileName_in, 'r' ); f2 = open( tpFileName, 'w' );
  for line in f1 :
    lineout = line.replace( '(FinalBSplineInterpolationOrder 3)', '(FinalBSplineInterpolationOrder 0)' );
    lineout = re.sub( "(ResultImageFormat \"mhd\")", "ResultImageFormat \"mha\"", lineout );
    lineout = re.sub( "(ResultImagePixelType \"short\")", "ResultImagePixelType \"unsigned char\"", lineout );
    lineout = re.sub( "(CompressResultImage \"false\")", "CompressResultImage \"true\"", lineout );
    f2.write( lineout );
  f1.close(); f2.close();

  # Transform the moving image segmentation to mimick the baseline result
  seg = os.path.join( options.directory, "result.mha" );
  seg_defm = os.path.join( options.directory, "segmentation_deformed.mha" );
  subprocess.call( [ "transformix", "-in", options.mseg, "-out", options.directory, "-tp", tpFileName ],
    stdout=subprocess.PIPE );
  if( os.path.exists( seg_defm ) ) : os.remove( seg_defm );
  shutil.move( seg, seg_defm );

  #
  # Deform the moving image segmentation by the baseline result
  #
  print( "Deforming moving image segmentation using " + tpFileName_b_in );

  # Make the transform parameters file suitable for binary images
  f1 = open( tpFileName_b_in, 'r' ); f2 = open( tpFileName_b, 'w' );
  for line in f1 :
    lineout = line.replace( '(FinalBSplineInterpolationOrder 3)', '(FinalBSplineInterpolationOrder 0)' );
    lineout = re.sub( "(ResultImageFormat \"mhd\")", "ResultImageFormat \"mha\"", lineout );
    lineout = re.sub( "(ResultImagePixelType \"short\")", "ResultImagePixelType \"unsigned char\"", lineout );
    lineout = re.sub( "(CompressResultImage \"false\")", "CompressResultImage \"true\"", lineout );
    f2.write( lineout );
  f1.close(); f2.close();

  # Transform the moving image segmentation to mimick the fixed image segmentation
  seg_defb = os.path.join( options.directory, "segmentation_baseline.mha" );
  subprocess.call( [ "transformix", "-in", options.mseg, "-out", options.directory, "-tp", tpFileName_b ],
    stdout=subprocess.PIPE );
  if( os.path.exists( seg_defb ) ) : os.remove( seg_defb );
  shutil.move( seg, seg_defb );

  # Compute the overlap between baseline segmentation and deformed moving segmentation
  try :
    # This will work from python 2.7 on
    outputAsString = subprocess.check_output( [ "elxComputeOverlap", "-in", seg_defm, seg_defb ] ).decode("utf-8");
  except :
    # Workaround for python 2.6 and lower. For MacMini specifically.
    outputAsString = subprocess.Popen( [ "elxComputeOverlap", "-in", seg_defm, seg_defb ], stdout=subprocess.PIPE ).communicate()[0].decode("utf-8");
  overlap = outputAsString[ outputAsString.find( "Overlap" ) : ].strip( "Overlap: " );

  # Report
  print( "The segmentation overlap between current and baseline is " + overlap );
  if float( overlap ) > 0.99 :
    print( "SUCCESS: overlap is higher than 0.99" );
    return 0;
  else :
    print( "FAILURE: overlap is lower than 0.99" );
    return 1;

#-------------------------------------------------------------------------------
if __name__ == '__main__':
    sys.exit(main())
