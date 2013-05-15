/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __transformixlib_CXX_
#define __transformixlib_CXX_

#include "transformixlib.h"

#ifdef _ELASTIX_USE_MEVISDICOMTIFF
  #include "itkUseMevisDicomTiff.h"
#endif

#include "elxTransformixMain.h"
#include <iostream>
#include <string>
#include <vector>
#include <queue>

#include "itkObject.h"
#include "itkDataObject.h"
#include <itksys/SystemTools.hxx>
#include <itksys/SystemInformation.hxx>

#include "elxTimer.h"

namespace transformix
{
/**
 * ******************* Constructor ***********************
 */

TRANSFORMIX::TRANSFORMIX()
{
  this->m_ResultImage = 0;
} // end Constructor


/**
 * ******************* Destructor ***********************
 */

TRANSFORMIX::~TRANSFORMIX()
{
  this->m_ResultImage = 0;
} // end Destructor


/**
 * ******************* Destructor ***********************
 */

TRANSFORMIX::ImagePointer
TRANSFORMIX::GetResultImage( void )
{
  return this->m_ResultImage;
}


/**
 * ******************* TransformImage ***********************
 */

int 
TRANSFORMIX::TransformImage( 
  ImagePointer inputImage,
  ParameterMapType & parameterMap,
  std::string outputPath,
  bool performLogging,
  bool performCout )
{
  /** Some typedef's.*/
  typedef elx::TransformixMain                        TransformixMainType;
  typedef TransformixMainType::Pointer                TransformixMainPointer;
  typedef TransformixMainType::ArgumentMapType        ArgumentMapType;
  typedef ArgumentMapType::value_type                 ArgumentMapEntryType;
  typedef elx::ElastixMain                            ElastixMainType;
  typedef ElastixMainType::DataObjectContainerType    DataObjectContainerType;
  typedef ElastixMainType::DataObjectContainerPointer DataObjectContainerPointer;

  /** Declare an instance of the Transformix class. */
  TransformixMainPointer transformix;

  DataObjectContainerPointer movingImageContainer = 0;
  DataObjectContainerPointer ResultImageContainer = 0;
  
  /** Initialize. */
  int               returndummy = 0;
  ArgumentMapType   argMap;
  bool              outFolderPresent = false;
  std::string       outFolder = "";
  std::string       logFileName = "";
 
  std::string key;
  std::string value;
  
  if( !outputPath.empty() )
  {
    key = "-out";
    value = outputPath;

    /** Make sure that last character of the output folder equals a '/'. */
    if( value.find_last_of( "/" ) != value.size() -1  )
    {
      value.append("/"); 
    }

    outFolderPresent = true;
  }
  else
  {
    /** Put command line parameters into parameterFileList. */
    //there must be an "-out", this is checked later in code!!
    key = "-out";
    value = "output_path_not_set";
  }

  /** Save this information. */
  outFolder = value;

  /** Attempt to save the arguments in the ArgumentMap. */
  if( argMap.count( key ) == 0 )
  {
    argMap.insert( ArgumentMapEntryType( key.c_str(), value.c_str() ) );
  }
  else if( performCout )
  {
    /** Duplicate arguments. */
    std::cerr << "WARNING!" << std::endl;
    std::cerr << "Argument "<< key.c_str() << "is only required once." << std::endl;
    std::cerr << "Arguments " << key.c_str() << " " << value.c_str() << "are ignored" << std::endl;
  }

  if( performLogging )
  {
    /** Check if the output directory exists. */
    bool outFolderExists = itksys::SystemTools::FileIsDirectory( outFolder.c_str() );
    if( !outFolderExists )
    {
      if( performCout )
      {
        std::cerr << "ERROR: the output directory does not exist." << std::endl;
        std::cerr << "You are responsible for creating it." << std::endl;
      }
      return( -2 );
    }
    else
    { 
      /** Setup xout. */
      if( performLogging )
      {
        logFileName = outFolder + "transformix.log";
      }
    }
  }

  /** The argv0 argument, required for finding the component.dll/so's. */
  argMap.insert( ArgumentMapEntryType( "-argv0", "transformix" ) );

  /** Setup xout. */
  int returndummy2 = elx::xoutSetup( logFileName.c_str() , performLogging , performCout );
  if( returndummy2 && performCout )
  {
    if( performCout )
    {
      std::cerr << "ERROR while setting up xout." << std::endl;
    }
    return( returndummy2 );
  }
  elxout << std::endl;

  /** Declare a timer, start it and print the start time. */
  tmr::Timer::Pointer totaltimer = tmr::Timer::New();
  totaltimer->StartTimer();
  elxout << "transformix is started at " << totaltimer->PrintStartTime()
    << ".\n" << std::endl;

  /**
   * ********************* START TRANSFORMATION *******************
   */

  /** Set transformix. */
  transformix = TransformixMainType::New();

  /** Set stuff from input or needed for output */
  movingImageContainer = DataObjectContainerType::New();
  movingImageContainer->CreateElementAt( 0 ) = inputImage;
  transformix->SetMovingImageContainer( movingImageContainer );
  transformix->SetResultImageContainer( ResultImageContainer ); 

  /** Run transformix. */
  returndummy = transformix->Run( argMap , parameterMap );

  /** Check if transformix run without errors. */
  if ( returndummy != 0 )
  {
    xl::xout["error"] << "Errors occurred" << std::endl;
    return returndummy;
  }

  /** Get the result image */
  ResultImageContainer = transformix->GetResultImageContainer(); 
  
  /** Stop timer and print it. */
  totaltimer->StopTimer();
  elxout << "\nTransformix has finished at " <<
    totaltimer->PrintStopTime() << "." << std::endl;
  elxout << "Elapsed time: " <<
    totaltimer->PrintElapsedTimeDHMS() << ".\n" << std::endl;

  this->m_ResultImage = ResultImageContainer->ElementAt( 0 );
  
  /** Clean up. */
  transformix = 0;
  TransformixMainType::UnloadComponents();

  /** Exit and return the error code. */
  return returndummy;

} // end TransformImage()

} // namespace transformix

#endif // end #ifndef __transformixlib_CXX_
