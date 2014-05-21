import sys
import os
import os.path
import fnmatch
import shutil
import glob
import itertools
import subprocess

#-------------------------------------------------------------------------------
# the main function
def main() :

  # The path to the source files relative to this script
  srcdir = os.path.join( "..", "src" );
  doxdir = os.path.join( "..", "dox" );
  toolsdir = os.path.join( "..", "tools" );

  # Get a list of all files
  patterns = ( '*.h', '*.hxx', '*.cxx', '*.cuh', '*.cu', '*.cl', '*.in', '*.txt', '*.tex', '*.bib', '*.cmake', '*.mhd', '*.html', '*.css', '*.dox', '*.py' );
  matches = [];
  for pattern in patterns :
    for root, dirnames, filenames in itertools.chain( os.walk( srcdir ), os.walk( doxdir ), os.walk( toolsdir ) ) :
      for filename in fnmatch.filter( filenames, pattern ) :
        matches.append( os.path.join( root, filename ) );

  for filename in matches :
    # Print the current file name
    print( filename );

    # Set the svn property
    #svn propset svn:eol-style native filename
    p = subprocess.Popen(
      "svn propset svn:eol-style native " + filename, stdout=subprocess.PIPE, shell=True );
    (output, err) = p.communicate();

  # Exit
  return 0

#-------------------------------------------------------------------------------
if __name__ == '__main__':
    sys.exit(main())
