import sys
import os
import os.path
import fnmatch
import shutil
import glob


#-------------------------------------------------------------------------------
# the main function
# cd tools
# python elxReplaceCopyrightNotice.py
def main() :

  # The path to the source files relative to this script
  srcdir = os.path.join( "..", "src" );

  # Get a list of all files
  #exclude = set( [ os.path.join( "..", "src", "Testing", "Baselines" ) ] );
  exclude = set( [ "Baselines", "ann_1.1", "CMake", "Data" ] );
  patterns = ( '*.h', '*.hxx', '*.cxx', '*.cuh', '*.cu', '*.in' );
  matches = [];
  for pattern in patterns :
    for root, dirnames, filenames in os.walk( srcdir ) :
      dirnames[:] = [ d for d in dirnames if d not in exclude ];
      for filename in fnmatch.filter( filenames, pattern ) :
        matches.append( os.path.join( root, filename ) );

  # Read the new copyright notice from file
  cnotice = "CopyrightNotice_Apache.txt";
  needle = open( cnotice, 'rU' ).read().strip();

  for match in matches :
    # Read the current file
    inp = open( match, 'rU' );
    fileAsString = inp.read();
    inp.close();

    #print( fileAsString );
    if not fileAsString.startswith( needle ) :
      print( match + "    - copyright notice NOT found" );

  # Exit
  return 0

#-------------------------------------------------------------------------------
if __name__ == '__main__':
    sys.exit(main())
