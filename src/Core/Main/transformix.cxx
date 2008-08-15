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


int main( int argc, char **argv )
{
  /** Check if "-help" or "--version" was asked for.*/
  if ( argc == 1 )
  {
    std::cout << "Use \"transformix --help\" for information about transformix-usage." << std::endl;
    return 0;
  }
  else if ( argc == 2 )
  {
    std::string argument( argv[ 1 ] );
    if ( argument == "-help" || argument == "--help" )
    {
      PrintHelp();
      return 0;
    }
    else if( argument == "--version" )
    {
      std::cout << std::fixed;
      std::cout << std::showpoint;
      std::cout << std::setprecision(3);
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
  typedef elx::TransformixMain                      TransformixMainType;
  typedef TransformixMainType::Pointer              TransformixMainPointer;
  typedef TransformixMainType::ArgumentMapType      ArgumentMapType;
  typedef ArgumentMapType::value_type               ArgumentMapEntryType;

  /** Declare an instance of the Transformix class. */
  TransformixMainPointer  transformix;

  /** Initialize. */
  int               returndummy = 0;
  ArgumentMapType   argMap;
  bool              outFolderPresent = false;
  std::string       outFolder = "";
  std::string       logFileName = "";


  /** Put command line parameters into parameterFileList. */
  for ( unsigned int i = 1; i < argc - 1; i += 2 )
  { 
    std::string key( argv[ i ] );
    std::string value( argv[ i + 1 ] );
  
    if ( key == "-out" )
    {
      /** Make sure that last character of the outputfolder equals a '/'. */
      if ( value.find_last_of( "/" ) != value.size() - 1 )
      {
        value.append( "/" );
      }

      /** Save this information. */
      outFolderPresent = true;
      outFolder = value;

    } // end if key == "-out"

    /** Attempt to save the arguments in the ArgumentMap. */
    if ( argMap.count( key ) == 0 )
    { 
      argMap.insert( ArgumentMapEntryType( key.c_str(), value.c_str() ) );
    }
    else
    {
      /** Duplicate arguments. */
      std::cerr << "WARNING!" << std::endl;
      std::cerr << "Argument "<< key.c_str() << "is only required once." << std::endl;
      std::cerr << "Arguments " << key.c_str() << " " << value.c_str() << "are ignored" << std::endl;
    }

  } // end for loop

  /** The argv0 argument, required for finding the component.dll/so's. */
  argMap.insert( ArgumentMapEntryType( "-argv0", argv[ 0 ] )  );

  /** Check that the option "-tp" is given. */
  if ( argMap.count( "-tp" ) == 0 )
  {
    std::cerr << "ERROR: No CommandLine option \"-tp\" given!" << std::endl;
    returndummy |= -1;
  }

  /** Check that at least one of the options "-in" or "-ipp" is given. */
  if ( argMap.count( "-in" ) == 0 && argMap.count( "-ipp" ) == 0 )
  {
    std::cerr << "ERROR: At least one of the CommandLine options \"-in\" pr \"-ipp\" should be given!" << std::endl;
    returndummy |= -1;
  }

  /** Check if the -out option is given and setup xout. */
  if ( outFolderPresent )
  {
    /** Check if the output directory exists. */
    bool outFolderExists = itksys::SystemTools::FileIsDirectory( outFolder.c_str() );
    if ( !outFolderExists )
    {
      std::cerr << "ERROR: the output directory does not exist." << std::endl;
      std::cerr << "You are responsible for creating it." << std::endl;
      returndummy |= -2;
    }
    else
    {
      /** Setup xout. */
      logFileName = argMap[ "-out" ] + "transformix.log" ;
      int returndummy2 = elx::xoutSetup( logFileName.c_str() );
      if ( returndummy2 )
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

  /** Stop if some fatal errors occured. */
  if ( returndummy )
  {
    return returndummy;
  }

  elxout << std::endl;

  /** Declare a timer, start it and print the start time. */
  tmr::Timer::Pointer totaltimer = tmr::Timer::New();
  totaltimer->StartTimer();
  elxout << "Transformix is started at " <<
    totaltimer->PrintStartTime() << ".\n" << std::endl;

  /**
   * ********************* START TRANSFORMATION *******************
   */

  /** Set transformix. */
  transformix = TransformixMainType::New();
  
  /** Print a start message. */
  elxout << "Running Transformix with parameter file \""
      << argMap[ "-tp" ] << "\".\n" << std::endl;

  /** Run transformix. */
  returndummy = transformix->Run( argMap );
  
  /** Check if runned without errors. */
  if ( returndummy != 0 )
  {
    xl::xout["error"] << "Errors occured" << std::endl;
    return returndummy;
  }

  /** Stop timer and print it. */
  totaltimer->StopTimer();
  elxout << "\nTransformix has finished at " <<
    totaltimer->PrintStopTime() << "." << std::endl;
  elxout << "Elapsed time: " <<
    totaltimer->PrintElapsedTimeDHMS() << ".\n" << std::endl;

  /** Clean up. */
  transformix = 0;
  TransformixMainType::UnloadComponents();

  /** Exit and return the error code. */
  return returndummy;

} // end main


/**
 * *********************** PrintHelp ****************************
 */

void PrintHelp(void)
{
  std::cout << "*********** transformix help: ***********\n\n";

  /** What is transformix? */
  std::cout << "Transformix applies a transform on an input image." << std::endl;
  std::cout << "The transform is specified in the transform-parameter file."
    << std::endl << std::endl;

  /** Mandatory argments. */
  std::cout << "Call transformix from the command line with mandatory arguments:" << std::endl;
  std::cout << "-out      output directory" << std::endl;
  std::cout << "-tp       transform-parameter file, only 1" << std::endl << std::endl;

  /** Optional arguments. */
  std::cout << "Optional extra commands:" << std::endl;
  std::cout << "-in       input image to deform" << std::endl;
  std::cout << "-ipp      file containing input-image points" << std::endl;
  std::cout << "          the point are transformed according to the specified transform-parameter file" << std::endl;
  std::cout << "          use \"-ipp all\" to transform all points from the input-image" << std::endl;
  std::cout << "-priority set the process priority to high or belownormal (Windows only)" << std::endl;
  std::cout << "-threads  set the maximum number of threads of transformix" << std::endl;
  std::cout << "At least one of the options \"-in\" or \"-ipp\" should be given."
    << std::endl << std::endl;
  
  /** The parameter file. */
  std::cout << "The transform-parameter file must contain all the information necessary for transformix to run properly. That includes which transform to use, with which parameters, etc." << std::endl;
  std::cout << "For a usable transform-parameter file, see the output of elastix." << std::endl << std::endl;

  std::cout << "Need further help? Check the website http://elastix.isi.uu.nl, or ask Marius and/or Stefan. :-)" << std::endl;

} // end PrintHelp()


#endif // end #ifndef __transformix_CXX_

