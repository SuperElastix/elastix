import sys
import os
import os.path
import fnmatch
import shutil
import glob
import subprocess
import re

#-------------------------------------------------------------------------------
# the main function
def main() :

  # The path to the source files relative to this script
  srcdir = os.path.join( "..", "src" );

  # Get a list of all files
  patterns = ( 'CMakeLists.txt', '*.cmake' );
  matches = [];
  for pattern in patterns :
    for root, dirnames, filenames in os.walk( srcdir ):
      for filename in fnmatch.filter( filenames, pattern ) :
        matches.append( os.path.join( root, filename ) );
  #print( matches );

  # Create a list of cmake commands (in lower case)
  # cmake --help-command-list >
  commandlistfilename = "commandlist.txt";
  commandlist = open( commandlistfilename, "w" );
  p = subprocess.Popen(
    "cmake --help-command-list", stdout=commandlist, shell=True );
  (output, err) = p.communicate();
  commandlist.close( );
  # Remove first line would be nice ...

  # Read lower case commands in a list
  commands = open( commandlistfilename ).read().splitlines();

  for match in matches :
    # Print the current file name
    print( match );
    #continue;

    # Read the current file as a string
    inp1 = open( match, 'r' );
    fileAsString = inp1.read();
    inp1.close();

    # Replace all entries from the dictionary
    for i in commands :
      ii = r'\b' + re.escape( i ) + r'\b\s*\(';
      re_i = re.compile( ii, re.IGNORECASE );
      fileAsString = re_i.sub( i + r'(', fileAsString );

    # Replace endif(...) with endif()
    re_endif = re.compile( r'endif\(.*\)' );
    fileAsString = re_endif.sub( r'endif()', fileAsString );

    # Overwrite
    inp2 = open( match, 'w' );
    inp2.write( fileAsString );
    inp2.close();

  # Clean up
  if os.path.exists( commandlistfilename ) : os.remove( commandlistfilename );

  # Exit
  return 0

#-------------------------------------------------------------------------------
if __name__ == '__main__':
    sys.exit(main())
