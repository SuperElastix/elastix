import sys
import os
import os.path
import platform
import shutil
from optparse import OptionParser

# -------------------------------------------------------------------------------
# the main function
#
# This python script applies uncrustify (Code Beautifier) to the elastix code.
#
# Uncrustify web site: http://uncrustify.sourceforge.net/
# Compile and add uncrustify to your $PATH environment variable.
#
# The configuration file for uncrustify is called elastixUncrustify.cfg.
#
# Execute this script without any option to create a preview directory
# called '_beautiful_code' in the src\_beautiful_code. 
# To apply uncrustify directly to the code use option '-a'.
# Including and excluding files and directories could be also indicated, see options.
#
# The script should be executed from the tools directory.
#
# \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
# Department of Radiology, Leiden, The Netherlands
#
# \note This work was funded by the Netherlands Organisation for
# Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).

def main():
    # usage, parse parameters
    usage = "usage: %prog [options] arg";
    parser = OptionParser(usage);

    # option to debug and verbose
    parser.add_option("-d", "--debug", action="store_true", dest="debug",
                      default=False, help="debug commands calls");
    parser.add_option("-v", "--verbose", action="store_true",
                      default=False, dest="verbose", help="verbose");
    parser.add_option("-q", "--quiet", action="store_true",
                      default=False, dest="quiet", help="quiet mode for uncrustify");

    # user defined uncrustify configuration file
    parser.add_option("-c", "--config", dest="config",
                      type="string", help="uncrustify configuration file");

    # options to control files. use -o output1 not -o c:\\temp
    parser.add_option("-o", "--output", dest="output_directory",
                      default="_beautiful_code", help="relative uncrustify output directory");

    # include regex, use syntax -i "value1 value2" NOT -i value1 value2
    parser.add_option("-i", "--include-regex", dest="include",
                      type="string", help="include files matching regular expression");

    # exclude regex, use syntax -e "value1 value2" NOT -e value1 value2
    parser.add_option("-e", "--exclude-regex", dest="exclude",
                      default="Common/KNN/ann_1.1/ Common/CUDA/",
                      help="exclude files matching regular expression");

    # apply directly to the code, has to be confirmed
    parser.add_option("-a", "--apply", action="store_true", default=False,
                      dest="apply", help="apply uncrustify directly to the svn files");

    (options, args) = parser.parse_args();

    # check if option -a is given, and confirm
    if options.apply == True:
        answer = query_yes_no("WARNING: Do you want to apply Uncrustify directly to the code?")
        if answer == False:
            return 0

    # get include and exclude options as lists
    include_list = None;
    exclude_list = None;
    if options.include != None:
        include_list = options.include.split(" ");
    if options.exclude != None:
        exclude_list = options.exclude.split(" ");

    # check for contradictory options
    if options.include != None and options.exclude != None:
        for include in include_list:
            for exclude in exclude_list:
                if include == exclude:
                    print( "ERROR: The contradictory options provided -i " + include + " -e " + exclude + "." );
                    return 1;

    # list of valid C++ extensions and top directories
    src_valid_cxx_extensions = set([".h", ".cpp", ".cxx", ".hxx", ".txx"]);  # not .in.h
    src_top_dirs = ["Common", "Components", "Core", "Testing"];

    # Define the current working directory and the directory containing this script
    current_dir = os.getcwd();
    current_dir = os.path.abspath(os.path.join(current_dir, '../src'));
    current_dir = current_dir.replace("/cygdrive/c/", "C:/");  # Cygwin support
    current_dir = current_dir.replace("/cygdrive/d/", "D:/");  # Cygwin support
    script_dir = os.path.dirname(os.path.realpath(__file__));
    script_dir = script_dir.replace("/cygdrive/c/", "C:/");  # Cygwin support
    script_dir = script_dir.replace("/cygdrive/d/", "D:/");  # Cygwin support

    # Find the uncrustify configuration file
    #
    if options.config == None:
        # If no configuration file is given via the options, then we look for a default
        # First we look in the directory of this script
        app_uncrustify_cfg_file = os.path.join(script_dir, "elastixUncrustify.cfg");
        # If it doesn't exist we look in the current working directory
        if not os.path.exists(app_uncrustify_cfg_file):
            app_uncrustify_cfg_file = os.path.join(current_dir, "elastixUncrustify.cfg");
        # If it doesn't exist we throw an error
        if not os.path.exists(app_uncrustify_cfg_file):
            print( "ERROR: cannot find the configuration file elastixUncrustify.cfg" );
            print( "  looked in: " + script_dir );
            print( "  looked in: " + current_dir );
            return 1;
    else:
        # Otherwise we use the user supplied configuration file
        if not os.path.exists(options.config):
            print( "ERROR: The configuration file '" + options.config + "' does not exist." );
            return 1;

        app_uncrustify_cfg_file = os.path.realpath(options.config);
        app_uncrustify_cfg_file = app_uncrustify_cfg_file.replace("/cygdrive/c/", "C:/");  # Cygwin support
        app_uncrustify_cfg_file = app_uncrustify_cfg_file.replace("/cygdrive/d/", "D:/");  # Cygwin support

    if options.verbose == True:
        print( "DEBUG: uncrustify configuration file: " + app_uncrustify_cfg_file );

    # uncrustify executable
    uncrustify_exe_name = "uncrustify";

    # get system type and define uncrustify executable
    if get_system_name() == 'windows':
        uncrustify_exe_name += ".exe"

    # check that uncrustify executable exists
    if which(uncrustify_exe_name) == None:
        print( "ERROR: The executable file '" + uncrustify_exe_name + "' does not exist in your PATH variable." );
        return 1;

    # create files list for uncrustify and place it in the output directory
    output_dir = os.path.join(current_dir, options.output_directory);
    if options.apply == False:
        create_dir(output_dir, options);

    # files list for uncrustify option -F:
    # -F FILE: read files to process from FILE, one filename per line
    app_uncrustify_files_list = "uncrustify_files.txt";
    if options.apply == True:
        app_uncrustify_files_list = os.path.join(current_dir, app_uncrustify_files_list);
    else:
        app_uncrustify_files_list = os.path.join(output_dir, app_uncrustify_files_list);

    # create files list, it has to be defined in uncrustify specific way
    filelist = None;

    if options.verbose == True:
        print( "DEBUG: Opening file " + app_uncrustify_files_list );
    filelist = open(app_uncrustify_files_list, "w");

    for dir in src_top_dirs:
        local_current_dir = os.path.join(current_dir, dir);
        for root, dirs, files in os.walk(local_current_dir):
            for f in files:
                fullpath = os.path.join(root, f);
                if file_valid(fullpath, include_list, exclude_list, src_valid_cxx_extensions):
                    rfullpath = fullpath.replace(current_dir, '');
                    rfullpath = rfullpath.lstrip('\\');
                    rfullpath = rfullpath.lstrip('/');  # Windows support
                    if options.debug == True:
                        print( "DEBUG: " + rfullpath );
                    else:
                        filelist.write(rfullpath);
                        filelist.write('\n');

    if options.verbose == True:
        print( "DEBUG: Closing file " + app_uncrustify_files_list );
    filelist.close();

    # define uncrustify arguments
    arg = "-c " + app_uncrustify_cfg_file;
    if options.apply == True:
        arg = arg + " --no-backup";
    else:
        arg = arg + " --prefix=" + output_dir;
    arg = arg + " -F " + app_uncrustify_files_list;
    if options.quiet == True:
        arg = arg + " -q";

    # execute uncrustify
    os.chdir(current_dir);
    call_program(uncrustify_exe_name, arg, options);

    # delete filelist if option apply is used
    if os.access(app_uncrustify_files_list, os.F_OK) == True and options.apply == True:
        if options.verbose == True:
            print( "DEBUG: Deleting file " + app_uncrustify_files_list );
        os.remove(app_uncrustify_files_list);

    # report
    print( "SUCCESS! Your code is beautiful!" );
    if not options.apply:
        print( "Check the directory " + output_dir );

    return 0


# -------------------------------------------------------------------------------
# get_system_name
def get_system_name():
    # store the architecture string (heavy performance drain)
    current_system = platform.system();
    current_system = current_system.lower();
    if current_system == 'windows':
        return 'windows';
    elif current_system == 'linux':
        return 'linux';
    elif current_system == 'darwin':
        return 'linux';
    elif 'cygwin' in current_system:
        return 'linux';
    else:
        return current_system;


# -------------------------------------------------------------------------------
# query_yes_no
def query_yes_no(question, default="yes"):
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False};
    if default == None:
        prompt = " [y/n] ";
    elif default == "yes":
        prompt = " [Y/n] ";
    elif default == "no":
        prompt = " [y/N] ";
    else:
        raise ValueError("invalid default answer: '%s'" % default);

    while True:
        sys.stdout.write(question + prompt);

        # First one works in python2, last one in python3
        try:
            choice = raw_input().lower();
        except NameError:
            choice = input().lower();

        if default is not None and choice == '':
            print("1")
            return valid[default];
        elif choice in valid:
            print("2")
            return valid[choice];
        else:
            print("3")
            sys.stdout.write("Please respond with 'yes' or 'no' " \
                             "(or 'y' or 'n').\n");


# -------------------------------------------------------------------------------
# which
# Can be replaced by shutil.which() after python 3.3.
def which(program):
    import os

    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None


# -------------------------------------------------------------------------------
# create directory
def create_dir(dir_name, options):
    if os.access(dir_name, os.F_OK) == False:
        cmd = "DEBUG: Creating directory '%s'" % dir_name
        if options.debug == True:
            print( cmd );
        else:
            if options.verbose == True:
                print( cmd );
            os.mkdir(dir_name);


# -------------------------------------------------------------------------------
# file_valid
def file_valid(file, include_list, exclude_list, elx_extensions):
    # check exclude
    if exclude_list != None:
        for exclude in exclude_list:
            if file.find(exclude) != -1:
                return False;

    # check within include
    if include_list != None:
        for include in include_list:
            if file.find(include) != -1:
                for extension in elx_extensions:
                    if file.endswith(extension):
                        return True;
    else:
        for extension in elx_extensions:
            if file.endswith(extension):
                return True;

    return False;


# -------------------------------------------------------------------------------
# call_program
def call_program(app_name, app_arg, options):
    cmd = app_name;
    if len(app_arg) > 0:
        cmd = cmd + " " + app_arg;

    if options.verbose and options.debug == False:
        print( "Calling " + cmd );

    if options.debug == True:
        print( "DEBUG: Calling " + cmd );
    else:
        os.system(cmd);

#-------------------------------------------------------------------------------
if __name__ == '__main__':
    sys.exit(main())
