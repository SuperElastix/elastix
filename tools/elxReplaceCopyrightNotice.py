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
  patterns = ( '*.h', '*.hxx', '*.cxx', '*.cuh', '*.cu', '*.in' );
  matches = [];
  for pattern in patterns :
    for root, dirnames, filenames in os.walk( srcdir ):
      for filename in fnmatch.filter( filenames, pattern ) :
        matches.append( os.path.join( root, filename ) );

  # Read the old copyright notice from file
  noticeOld = "CopyrightNotice_BSD.txt";
  needle      = open( noticeOld ).read();
  #print( needle );

  # Read the new copyright notice from file
  noticeNew = "CopyrightNotice_Apache.txt";
  replacement = open( noticeNew ).read();
  #print( replacement );

  for match in matches :
    # Print the current file name
    print( match );
    #continue;

    # Read the current file
    inp1 = open( match, 'r' );
    fileAsString = inp1.read();
    inp1.close();

    # Replace the copyright notice
    inp2 = open( match, 'w' );
    inp2.write( fileAsString.replace( needle, replacement ) );
    inp2.close();

  # Exit
  return 0

#-------------------------------------------------------------------------------
if __name__ == '__main__':
    sys.exit(main())
