import sys
import os
import os.path
import subprocess
import shutil
import platform
import argparse


#-------------------------------------------------------------------------------
#
# This python script runs nightly dashboards for elastix.
#
# To circumvent all kinds of problems, the latest nightly dashboard scripts
# are checked out to a temporary directory, after which they are called.
# After running the dashboard these temporary scripts are deleted again.
#
# This python script assumes that svn, git and ctest can be found on the PATH.
#
def main():

  # Create an argument parser
  parser = argparse.ArgumentParser()

  # Argument to select the dashboard script (required)
  # e.g. -s elxDashboard_LKEB_win10-64_VS2015.cmake
  parser.add_argument( "-s", "--script",
    dest='dashboard_script', required=True,
    choices = [
      'elxDashboard_LKEB_win7-64_VS2010.cmake',
      'elxDashboard_LKEB_win10-64_VS2015.cmake',
      'elxDashboard_LKEB_linux64_gcc_Debug.cmake',
      'elxDashboard_LKEB_linux64_gcc_Release.cmake',
      'elxDashboard_LKEB_linux64_clang_Debug.cmake',
      'elxDashboard_LKEB_linux64_clang_Release.cmake',
      'elxDashboard_LKEB_macosx64_gcc.cmake',
      'elxDashboard_BIGR_linux64_gcc.cmake',
      'elxDashboard_BIGR_winXP32_VS2008.cmake' ],
    help='select the dashboard to run' );

  # Argument to select the compilation mode (optional)
  # e.g. -C Debug
  parser.add_argument( "-C", default='Release', dest='C',
    choices = [ 'Debug', 'Release' ],
    help='select the compilation mode' );

  # Argument to select the local working directory (required)
  # e.g. -d "D:/toolkits/elastix/nightly"
  # e.g. -d  "/home/marius/nightly-builds"
  parser.add_argument( "-d", "--dir",
    dest='local', required=True,
    help='set the local working directory' );

  # Argument to select the elastix remote (optional)
  # By default it points to https://github.com/SuperElastix/elastix
  # but this option can be used to select user-specific forks.
  # e.g. -r https://github.com/SuperElastix/elastix
  # e.g. -r https://github.com/mstaring/elastix
  #parser.add_argument( "-r", "--remote",
  #  default='https://github.com/SuperElastix/elastix', dest='remote_base',
  #  help='select elastix remote' );
  # This is currently hard-coded in elxDashboardCommon.cmake,
  # and cannot be an option here atm.
  remote_base = "https://github.com/SuperElastix/elastix";

  # Argument to select the elastix branch (optional)
  # e.g. -b performance_ITK4
  #parser.add_argument( "-b", "--branch",
  #  default='develop', dest='branch',
  #  help='select elastix branch' );
  # This is currently hard-coded in elxDashboardCommon.cmake,
  # and cannot be an option here atm.
  branch = "develop";

  options = parser.parse_args();

  # Define and open log file
  tmpname = options.dashboard_script.rstrip( '.cmake' ).lstrip( 'elxDashboard_' );
  log  = options.local + "/nightlybuild_" + tmpname + ".log";
  flog = open( log, "w" );

  # Check if required executables can be found
  for exe in [ 'svn', 'git', 'ctest' ] :
    # Since whutil.which() only exists from python 3.3 we use try
    try :
      if shutil.which( exe ) == None :
        message = "ERROR: unable to find executable: " + exe;
        print( message );
        flog.write( message );
        exit();
    except : pass;

  # Remote. Define the remote location of the dashboard scripts
  remote        = remote_base + "/branches/" + branch + "/Testing/Dashboard";
  remote_common = remote + "/elxDashboardCommon.cmake";
  remote_dash   = remote + "/" + options.dashboard_script;

  # Local. Define the local nightly directory on this machine
  tmpdir        = options.local  + "/elx_" + tmpname + "_tmp";
  local_common  = tmpdir + "/elxDashboardCommon.cmake";
  local_dash    = tmpdir + "/" + options.dashboard_script;

  # Checkout the latest dashboard scripts
  #
  # The following relies on the GitHub support of Subversion.
  # It allows us to svn export only the dashboard files, and not
  # the complete elastix repository.
  #
  # elastix.git can be approached as follows:
  # svn export https://github.com/SuperElastix/elastix/trunk/path/to/file
  # for the main branch 'master'. Branches can be approached like:
  # svn export https://github.com/SuperElastix/elastix/branches/develop/path/to/file

  # Create local temporary directory to checkout dashboard scripts to
  shutil.rmtree( tmpdir, True );
  os.mkdir( tmpdir );

  # Export both the common and the specific dashboard files.
  command = [ "svn", "export", "--non-interactive", "--trust-server-cert" ];
  command.extend( [ remote_common, local_common ] );
  flog.write( ' '.join( command ) ); flog.write( '\n' );
  subprocess.call( command, stdout = flog, stderr = subprocess.STDOUT );

  command = [ "svn", "export", "--non-interactive", "--trust-server-cert" ];
  command.extend( [ remote_dash, local_dash ] );
  flog.write( ' '.join( command ) ); flog.write( '\n' );
  subprocess.call( command, stdout = flog, stderr = subprocess.STDOUT );

  flog.write( "Dashboard scripts checked out\n" );

  # Run the nightly dashboard!
  #
  command = [ "ctest", "-C", options.C, "-S", local_dash, "-V" ];
  flog.write( ' '.join( command ) ); flog.write( '\n' );
  subprocess.call( command, stdout = flog, stderr = subprocess.STDOUT );

  # Close log file
  flog.close();

  # Clean up, silence warnings
  if os.path.exists( tmpdir ) :
    shutil.rmtree( tmpdir, True );
  if os.path.exists( tmpdir ) :
    os.system( 'rmdir /S /Q \"' + tmpdir + '\"' );

#-------------------------------------------------------------------------------
if __name__ == '__main__' :
  sys.exit(main())
