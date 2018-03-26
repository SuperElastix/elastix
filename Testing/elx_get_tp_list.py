import sys
import os
import re
import os.path
import glob
import shutil
from optparse import OptionParser

#-------------------------------------------------------------------------------
# the main function
# cd bin_VS2010
# ctest -C Release
# cd Testing
# python ../../Testing/elx_get_tp_list.py -l elastix_run* -o ../../Testing/Baselines
# cd ..
# cmake .
# ctest -C Release -R COMPARE_TP
# cd ..
# git add Testing/Baselines/TransformParameters_*
# git commit -m "ENH: updating baselines after recent change X"
# git push
def main():
    # usage, parse parameters
    usage = "usage: %prog [options] arg";
    parser = OptionParser( usage );

    # option to debug and verbose
    parser.add_option( "-v", "--verbose",
      action="store_true", dest="verbose" );

    # options to control files
    parser.add_option( "-l", "--list", type="string", dest="directoryList", help="list of elastix output directories" );
    parser.add_option( "-o", "--output", type="string", dest="outputDirectory", help="output base directory" );

    (options, args) = parser.parse_args();

    # Check if option -l is given
    if options.directoryList == None :
      parser.error( "The option directory list (-l) should be given" );

    # Check if option -d is given
    if options.outputDirectory == None :
      parser.error( "The option output directory (-o) should be given" );

    if not os.path.exists( options.outputDirectory ) :
      print( "The output directory " + options.outputDirectory + " does not exist. Create it before running" );

    # Convert string -l to list
    dirList = options.directoryList.split( ' ' );
    # Add everything not processed
    dirList.extend( args );

    for directory in dirList :
      # Skip directories with "Threads" in the name
      if "Threads" in directory : continue;

      # Find the largest TransformParameters.?.txt
      inputFileNames = glob.glob( os.path.join( directory, "TransformParameters.?.txt" ) );
      inputFileNames.sort( reverse = True );
      inputFileName = inputFileNames[0];

      # Create a name for the output file name
      directory = re.sub( '\/$', '', directory );
      dir_part = list( os.path.split( directory ) ).pop();
      dir_part = dir_part.replace( "elastix_run_", "", 1 );
      outputFileName = "TransformParameters_" + dir_part + ".txt.in";
      outputFileName = os.path.join( options.outputDirectory, outputFileName );
      print( outputFileName );

      # Copy the results as the new baselines, while replacing
      # the initial transform to point to the correct path.
      f1 = open( inputFileName, 'rU' );
      f2 = open( outputFileName, 'w' );
      for oldline in f1 :
        newline = oldline;
        if "InitialTransformParametersFileName" in oldline :
          oldFileName = os.path.basename( oldline.split()[1][1:-2] );
          newFileName = os.path.join( "@ELASTIX_DATA_DIR@", oldFileName );
          if not "NoInitialTransform" in oldFileName :
            newline = "(InitialTransformParametersFileName \"" + newFileName + "\")\n";
        f2.write( newline );
      f1.close(); f2.close();

    return 0

#-------------------------------------------------------------------------------
if __name__ == '__main__':
    sys.exit(main())
