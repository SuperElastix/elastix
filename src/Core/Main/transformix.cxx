/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __transformix_CXX_
#define __transformix_CXX_

#include "transformix.h"

int
main( int argc, char ** argv )
{
  /** Check if "-help" or "--version" was asked for.*/
  if( argc == 1 )
  {
    std::cout << "Use \"transformix --help\" for information about transformix-usage." << std::endl;
    return 0;
  }
  else if( argc == 2 )
  {
    std::string argument( argv[ 1 ] );
    if( argument == "-help" || argument == "--help" || argument == "-h" )
    {
      PrintHelp();
      return 0;
    }
    else if( argument == "--version" )
    {
      std::cout << std::fixed;
      std::cout << std::showpoint;
      std::cout << std::setprecision( 3 );
      std::cout << "transformix version: " << __ELASTIX_VERSION << std::endl;
      return 0;
    }
    else
    {
      std::cout << "Use \"transformix --help\" for information about transformix-usage." << std::endl;
      return 0;
    }
  }

  /** Some typedef's.*/
  typedef elx::TransformixMain                 TransformixMainType;
  typedef TransformixMainType::Pointer         TransformixMainPointer;
  typedef TransformixMainType::ArgumentMapType ArgumentMapType;
  typedef ArgumentMapType::value_type          ArgumentMapEntryType;

  /** Support Mevis Dicom Tiff (if selected in cmake) */
  RegisterMevisDicomTiff();

  /** Declare an instance of the Transformix class. */
  TransformixMainPointer transformix;

  /** Initialize. */
  int             returndummy = 0;
  ArgumentMapType argMap;
  bool            outFolderPresent = false;
  std::string     outFolder        = "";
  std::string     logFileName      = "";

  /** Put command line parameters into parameterFileList. */
  for( unsigned int i = 1; static_cast< long >( i ) < argc - 1; i += 2 )
  {
    std::string key( argv[ i ] );
    std::string value( argv[ i + 1 ] );

    if( key == "-out" )
    {
      /** Make sure that last character of the output folder equals a '/' or '\'. */
      const char last = value[ value.size() - 1 ];
      if( last != '/' && last != '\\' ) { value.append( "/" ); }
      value = itksys::SystemTools::ConvertToOutputPath( value.c_str() );

      /** Note that on Windows, in case the output folder contains a space,
       * the path name is double quoted by ConvertToOutputPath, which is undesirable.
       * So, we remove these quotes again.
       */
      if( itksys::SystemTools::StringStartsWith( value.c_str(), "\"" )
        && itksys::SystemTools::StringEndsWith(   value.c_str(), "\"" ) )
      {
        value = value.substr( 1, value.length() - 2 );
      }

      /** Save this information. */
      outFolderPresent = true;
      outFolder        = value;

    } // end if key == "-out"

    /** Attempt to save the arguments in the ArgumentMap. */
    if( argMap.count( key ) == 0 )
    {
      argMap.insert( ArgumentMapEntryType( key.c_str(), value.c_str() ) );
    }
    else
    {
      /** Duplicate arguments. */
      std::cerr << "WARNING!" << std::endl;
      std::cerr << "Argument " << key.c_str() << "is only required once." << std::endl;
      std::cerr << "Arguments " << key.c_str() << " " << value.c_str() << "are ignored" << std::endl;
    }

  } // end for loop

  /** The argv0 argument, required for finding the component.dll/so's. */
  argMap.insert( ArgumentMapEntryType( "-argv0", argv[ 0 ] ) );

  /** Check that the option "-tp" is given. */
  if( argMap.count( "-tp" ) == 0 )
  {
    std::cerr << "ERROR: No CommandLine option \"-tp\" given!" << std::endl;
    returndummy |= -1;
  }

  /** Check that at least one of the following options is given. */
  if( argMap.count( "-in" ) == 0
    && argMap.count( "-ipp" ) == 0
    && argMap.count( "-def" ) == 0
    && argMap.count( "-jac" ) == 0
    && argMap.count( "-jacmat" ) == 0 )
  {
    std::cerr << "ERROR: At least one of the CommandLine options \"-in\", "
              << "\"-def\", \"-jac\", or \"-jacmat\" should be given!" << std::endl;
    returndummy |= -1;
  }

  /** Check if the -out option is given and setup xout. */
  if( outFolderPresent )
  {
    /** Check if the output directory exists. */
    bool outFolderExists = itksys::SystemTools::FileIsDirectory( outFolder.c_str() );
    if( !outFolderExists )
    {
      std::cerr << "ERROR: the output directory \"" << outFolder << "\" does not exist." << std::endl;
      std::cerr << "You are responsible for creating it." << std::endl;
      returndummy |= -2;
    }
    else
    {
      /** Setup xout. */
      logFileName = argMap[ "-out" ] + "transformix.log";
      int returndummy2 = elx::xoutSetup( logFileName.c_str(), true, true );
      if( returndummy2 )
      {
        std::cerr << "ERROR while setting up xout." << std::endl;
      }
      returndummy |= returndummy2;
    }
  }
  else
  {
    returndummy = -2;
    std::cerr << "ERROR: No CommandLine option \"-out\" given!" << std::endl;
  }

  /** Stop if some fatal errors occurred. */
  if( returndummy )
  {
    return returndummy;
  }

  elxout << std::endl;

  /** Declare a timer, start it and print the start time. */
  tmr::Timer::Pointer totaltimer = tmr::Timer::New();
  totaltimer->StartTimer();
  elxout << "transformix is started at " << totaltimer->PrintStartTime()
         << ".\n" << std::endl;

  /** Print where transformix was run. */
  elxout << "which transformix:   " << argv[ 0 ] << std::endl;
  itksys::SystemInformation info;
  info.RunCPUCheck();
  info.RunOSCheck();
  info.RunMemoryCheck();
  elxout << "transformix runs at: " << info.GetHostname() << std::endl;
  elxout << "  " << info.GetOSName() << " "
         << info.GetOSRelease() << ( info.Is64Bits() ? " (x64), " : ", " )
         << info.GetOSVersion() << std::endl;
  elxout << "  with " << info.GetTotalPhysicalMemory() << " MB memory, and "
         << info.GetNumberOfPhysicalCPU() << " cores @ "
         << static_cast< unsigned int >( info.GetProcessorClockFrequency() )
         << " MHz." << std::endl;

  /**
   * ********************* START TRANSFORMATION *******************
   */

  /** Set transformix. */
  transformix = TransformixMainType::New();

  /** Print a start message. */
  elxout << "Running transformix with parameter file \""
         << argMap[ "-tp" ] << "\".\n" << std::endl;

  /** Run transformix. */
  returndummy = transformix->Run( argMap );

  /** Check if transformix run without errors. */
  if( returndummy != 0 )
  {
    xl::xout[ "error" ] << "Errors occurred" << std::endl;
    return returndummy;
  }

  /** Stop timer and print it. */
  totaltimer->StopTimer();
  elxout << "\nTransformix has finished at "
         << totaltimer->PrintStopTime() << "." << std::endl;
  elxout << "Elapsed time: "
         << totaltimer->PrintElapsedTimeDHMS() << ".\n" << std::endl;

  /** Clean up. */
  transformix = 0;
  TransformixMainType::UnloadComponents();

  /** Exit and return the error code. */
  return returndummy;

} // end main


/**
 * *********************** PrintHelp ****************************
 */

void
PrintHelp( void )
{
  /** Print the version. */
  std::cout << std::fixed;
  std::cout << std::showpoint;
  std::cout << std::setprecision( 3 );
  std::cout << "transformix version: " << __ELASTIX_VERSION << "\n" << std::endl;

  /** What is transformix? */
  std::cout << "transformix applies a transform on an input image and/or "
    "generates a deformation field.\n";
  std::cout << "The transform is specified in the transform-parameter file.\n";
  std::cout << "  --help, -h displays this message and exit\n";
  std::cout << "  --version  output version information and exit\n" << std::endl;

  /** Mandatory arguments. */
  std::cout << "Call transformix from the command line with mandatory arguments:\n";
  std::cout << "  -out      output directory\n";
  std::cout << "  -tp       transform-parameter file, only 1\n" << std::endl;

  /** Optional arguments. */
  std::cout << "Optional extra commands:\n";
  std::cout << "  -in       input image to deform\n";
  std::cout << "  -def      file containing input-image points; the point are transformed\n"
            << "            according to the specified transform-parameter file\n";
  std::cout << "            use \"-def all\" to transform all points from the input-image, which\n"
            << "            effectively generates a deformation field.\n";
  std::cout << "  -jac      use \"-jac all\" to generate an image with the determinant of the\n"
            << "            spatial Jacobian\n";
  std::cout << "  -jacmat   use \"-jacmat all\" to generate an image with the spatial Jacobian\n"
            << "            matrix at each voxel\n";
  std::cout << "  -priority set the process priority to high, abovenormal, normal (default),\n"
            << "            belownormal, or idle (Windows only option)\n";
  std::cout << "  -threads  set the maximum number of threads of transformix\n";
  std::cout << "\nAt least one of the options \"-in\", \"-def\", \"-jac\", or \"-jacmat\" should be given.\n"
            << std::endl;

  /** The parameter file. */
  std::cout << "The transform-parameter file must contain all the information "
    "necessary for transformix to run properly. That includes which transform "
    "to use, with which parameters, etc. For a usable transform-parameter file, "
    "run elastix, and inspect the output file \"TransformParameters.0.txt\".\n" << std::endl;

  std::cout << "Need further help?\n"
    "Check the website http://elastix.isi.uu.nl, or mail elastix@bigr.nl." << std::endl;

} // end PrintHelp()


#endif // end #ifndef __transformix_CXX_
