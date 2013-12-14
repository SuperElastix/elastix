/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __elastix_cxx
#define __elastix_cxx

#include "elastix.h"

int
main( int argc, char ** argv )
{

  /** Check if "--help" or "--version" was asked for. */
  if( argc == 1 )
  {
    std::cout << "Use \"elastix --help\" for information about elastix-usage." << std::endl;
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
      std::cout << "elastix version: " << __ELASTIX_VERSION << std::endl;
      return 0;
    }
    else
    {
      std::cout << "Use \"elastix --help\" for information about elastix-usage." << std::endl;
      return 0;
    }
  }

  /** Some typedef's. */
  typedef elx::ElastixMain                            ElastixMainType;
  typedef ElastixMainType::Pointer                    ElastixMainPointer;
  typedef std::vector< ElastixMainPointer >           ElastixMainVectorType;
  typedef ElastixMainType::ObjectPointer              ObjectPointer;
  typedef ElastixMainType::DataObjectContainerPointer DataObjectContainerPointer;
  typedef ElastixMainType::FlatDirectionCosinesType   FlatDirectionCosinesType;

  typedef ElastixMainType::ArgumentMapType ArgumentMapType;
  typedef ArgumentMapType::value_type      ArgumentMapEntryType;

  typedef std::pair< std::string, std::string > ArgPairType;
  typedef std::queue< ArgPairType >             ParameterFileListType;
  typedef ParameterFileListType::value_type     ParameterFileListEntryType;

  /** Support Mevis Dicom Tiff (if selected in cmake) */
  RegisterMevisDicomTiff();

  /** Some declarations and initialisations. */
  ElastixMainVectorType elastices;

  ObjectPointer              transform            = 0;
  DataObjectContainerPointer fixedImageContainer  = 0;
  DataObjectContainerPointer movingImageContainer = 0;
  DataObjectContainerPointer fixedMaskContainer   = 0;
  DataObjectContainerPointer movingMaskContainer  = 0;
  FlatDirectionCosinesType   fixedImageOriginalDirection;
  int                        returndummy        = 0;
  unsigned long              nrOfParameterFiles = 0;
  ArgumentMapType            argMap;
  ParameterFileListType      parameterFileList;
  bool                       outFolderPresent = false;
  std::string                outFolder        = "";
  std::string                logFileName      = "";

  /** Put command line parameters into parameterFileList. */
  for( unsigned int i = 1; static_cast< long >( i ) < ( argc - 1 ); i += 2 )
  {
    std::string key( argv[ i ] );
    std::string value( argv[ i + 1 ] );

    if( key == "-p" )
    {
      /** Queue the ParameterFileNames. */
      nrOfParameterFiles++;
      parameterFileList.push(
        ParameterFileListEntryType( key.c_str(), value.c_str() ) );
      /** The different '-p' are stored in the argMap, with
       * keys p(1), p(2), etc. */
      std::ostringstream tempPname( "" );
      tempPname << "-p(" << nrOfParameterFiles << ")";
      std::string tempPName = tempPname.str();
      argMap.insert( ArgumentMapEntryType( tempPName.c_str(), value.c_str() ) );
    }
    else
    {
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
      if( argMap.count( key.c_str() ) == 0 )
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

    } // end else (so, if key does not equal "-p")

  } // end for loop

  /** The argv0 argument, required for finding the component.dll/so's. */
  argMap.insert( ArgumentMapEntryType( "-argv0", argv[ 0 ] ) );

  /** Check if at least once the option "-p" is given. */
  if( nrOfParameterFiles == 0 )
  {
    std::cerr << "ERROR: No CommandLine option \"-p\" given!" << std::endl;
    returndummy |= -1;
  }

  /** Check if the -out option is given. */
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
      logFileName = outFolder + "elastix.log";
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
  elxout << "elastix is started at " << totaltimer->PrintStartTime()
         << ".\n" << std::endl;

  /** Print where elastix was run. */
  elxout << "which elastix:   " << argv[ 0 ] << std::endl;
  itksys::SystemInformation info;
  info.RunCPUCheck();
  info.RunOSCheck();
  info.RunMemoryCheck();
  elxout << "elastix runs at: " << info.GetHostname() << std::endl;
  elxout << "  " << info.GetOSName() << " "
         << info.GetOSRelease() << ( info.Is64Bits() ? " (x64), " : ", " )
         << info.GetOSVersion() << std::endl;
  elxout << "  with " << info.GetTotalPhysicalMemory() << " MB memory, and "
         << info.GetNumberOfPhysicalCPU() << " cores @ "
         << static_cast< unsigned int >( info.GetProcessorClockFrequency() )
         << " MHz." << std::endl;

  /**
   * ********************* START REGISTRATION *********************
   *
   * Do the (possibly multiple) registration(s).
   */

  for( unsigned int i = 0; i < nrOfParameterFiles; i++ )
  {
    /** Create another instance of ElastixMain. */
    elastices.push_back( ElastixMainType::New() );

    /** Set stuff we get from a former registration. */
    elastices[ i ]->SetInitialTransform( transform );
    elastices[ i ]->SetFixedImageContainer( fixedImageContainer );
    elastices[ i ]->SetMovingImageContainer( movingImageContainer );
    elastices[ i ]->SetFixedMaskContainer( fixedMaskContainer );
    elastices[ i ]->SetMovingMaskContainer( movingMaskContainer );
    elastices[ i ]->SetOriginalFixedImageDirectionFlat( fixedImageOriginalDirection );

    /** Set the current elastix-level. */
    elastices[ i ]->SetElastixLevel( i );
    elastices[ i ]->SetTotalNumberOfElastixLevels( nrOfParameterFiles );

    /** Delete the previous ParameterFileName. */
    if( argMap.count( "-p" ) )
    {
      argMap.erase( "-p" );
    }

    /** Read the first parameterFileName in the queue. */
    ArgPairType argPair = parameterFileList.front();
    parameterFileList.pop();

    /** Put it in the ArgumentMap. */
    argMap.insert( ArgumentMapEntryType( argPair.first, argPair.second ) );

    /** Print a start message. */
    elxout << "-------------------------------------------------------------------------" << "\n" << std::endl;
    elxout << "Running elastix with parameter file " << i
           << ": \"" << argMap[ "-p" ] << "\".\n" << std::endl;

    /** Declare a timer, start it and print the start time. */
    tmr::Timer::Pointer timer = tmr::Timer::New();
    timer->StartTimer();
    elxout << "Current time: " << timer->PrintStartTime() << "." << std::endl;

    /** Start registration. */
    returndummy = elastices[ i ]->Run( argMap );

    /** Check for errors. */
    if( returndummy != 0 )
    {
      xl::xout[ "error" ] << "Errors occurred!" << std::endl;
      return returndummy;
    }

    /** Get the transform, the fixedImage and the movingImage
     * in order to put it in the (possibly) next registration.
     */
    transform                   = elastices[ i ]->GetFinalTransform();
    fixedImageContainer         = elastices[ i ]->GetFixedImageContainer();
    movingImageContainer        = elastices[ i ]->GetMovingImageContainer();
    fixedMaskContainer          = elastices[ i ]->GetFixedMaskContainer();
    movingMaskContainer         = elastices[ i ]->GetMovingMaskContainer();
    fixedImageOriginalDirection = elastices[ i ]->GetOriginalFixedImageDirectionFlat();

    /** Print a finish message. */
    elxout << "Running elastix with parameter file " << i
           << ": \"" << argMap[ "-p" ] << "\", has finished.\n" << std::endl;

    /** Stop timer and print it. */
    timer->StopTimer();
    elxout << "\nCurrent time: " << timer->PrintStopTime() << "." << std::endl;
    elxout << "Time used for running elastix with this parameter file: "
           << timer->PrintElapsedTimeDHMS() << ".\n" << std::endl;

    /** Try to release some memory. */
    elastices[ i ] = 0;

  } // end loop over registrations

  elxout << "-------------------------------------------------------------------------" << "\n" << std::endl;

  /** Stop totaltimer and print it. */
  totaltimer->StopTimer();
  elxout << "Total time elapsed: " << totaltimer->PrintElapsedTimeDHMS() << ".\n" << std::endl;

  /**
   * Make sure all the components that are defined in a Module (.DLL/.so)
   * are deleted before the modules are closed.
   */

  for( unsigned int i = 0; i < nrOfParameterFiles; i++ )
  {
    elastices[ i ] = 0;
  }

  transform            = 0;
  fixedImageContainer  = 0;
  movingImageContainer = 0;
  fixedMaskContainer   = 0;
  movingMaskContainer  = 0;

  /** Close the modules. */
  ElastixMainType::UnloadComponents();

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
  std::cout << "elastix version: " << __ELASTIX_VERSION << "\n" << std::endl;

  /** What is elastix? */
  std::cout << "elastix registers a moving image to a fixed image.\n";
  std::cout << "The registration-process is specified in the parameter file.\n";
  std::cout << "  --help, -h displays this message and exit\n";
  std::cout << "  --version  output version information and exit\n" << std::endl;

  /** Mandatory arguments.*/
  std::cout << "Call elastix from the command line with mandatory arguments:\n";
  std::cout << "  -f        fixed image\n";
  std::cout << "  -m        moving image\n";
  std::cout << "  -out      output directory\n";
  std::cout << "  -p        parameter file, elastix handles 1 or more \"-p\"\n"
            << std::endl;

  /** Optional arguments.*/
  std::cout << "Optional extra commands:\n";
  std::cout << "  -fMask    mask for fixed image\n";
  std::cout << "  -mMask    mask for moving image\n";
  std::cout << "  -t0       parameter file for initial transform\n";
  std::cout << "  -priority set the process priority to high, abovenormal, normal (default),\n"
            << "            belownormal, or idle (Windows only option)\n";
  std::cout << "  -threads  set the maximum number of threads of elastix\n"
            << std::endl;

  /** The parameter file.*/
  std::cout << "The parameter-file must contain all the information "
    "necessary for elastix to runproperly. That includes which metric to "
    "use, which optimizer, which transform, etc. It must also contain "
    "information specific for the metric, optimizer, transform, etc. "
    "For a usable parameter-file, see the website.\n" << std::endl;

  std::cout << "Need further help?\nCheck the website http://elastix.isi.uu.nl, "
    "or mail elastix@bigr.nl." << std::endl;

} // end PrintHelp()


#endif // end #ifndef __elastix_cxx
