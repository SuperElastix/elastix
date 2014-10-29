/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

// \todo: cxx's don't need ifdefs?
#ifndef __elastixlib_cxx
#define __elastixlib_cxx

#include "elastixlib.h"

#ifdef _ELASTIX_USE_MEVISDICOMTIFF
#include "itkUseMevisDicomTiff.h"
#endif

#include "elxElastixMain.h"
#include <iostream>
#include <string>
#include <vector>
#include <queue>
#include "itkObject.h"
#include "itkDataObject.h"
#include <itksys/SystemTools.hxx>
#include <itksys/SystemInformation.hxx>

#include "elxTimer.h"

namespace elastix
{

/**
 * ******************* Constructor ***********************
 */

ELASTIX::ELASTIX() :
  m_ResultImage( 0 )
{} // end Constructor

/**
 * ******************* Destructor ***********************
 */

ELASTIX::~ELASTIX()
{
  this->m_ResultImage = 0;
  this->m_TransformParametersList.clear();
} // end Destructor


/**
 * ******************* GetResultImage ***********************
 */

ELASTIX::ImagePointer
ELASTIX::GetResultImage( void )
{
  return this->m_ResultImage;
} // end GetResultImage()


/**
 * ******************* GetTransformParameterMap ***********************
 */

ELASTIX::ParameterMapType
ELASTIX::GetTransformParameterMap( void )
{
  return this->m_TransformParametersList[ this->m_TransformParametersList.size() - 1 ];
} // end GetTransformParameterMap()


/**
 * ******************* GetTransformParameterMapList ***********************
 */

ELASTIX::ParameterMapListType
ELASTIX::GetTransformParameterMapList( void )
{
  return this->m_TransformParametersList;
} // end GetTransformParameterMapList()


/**
 * ******************* RegisterImages ***********************
 */

int
ELASTIX::RegisterImages(
  ImagePointer fixedImage,
  ImagePointer movingImage,
  ParameterMapType & parameterMap,
  std::string outputPath,
  bool performLogging,
  bool performCout,
  ImagePointer fixedMask,
  ImagePointer movingMask )
{
  std::vector< ParameterMapType > parameterMaps( 1 );
  parameterMaps[ 0 ] = parameterMap;
  return this->RegisterImages(
    fixedImage, movingImage,
    parameterMaps,
    outputPath,
    performLogging, performCout,
    fixedMask, movingMask );

} // end RegisterImages()


/**
 * ******************* RegisterImages ***********************
 */

int
ELASTIX::RegisterImages(
  ImagePointer fixedImage,
  ImagePointer movingImage,
  std::vector< ParameterMapType > & parameterMaps,
  std::string outputPath,
  bool performLogging,
  bool performCout,
  ImagePointer fixedMask,
  ImagePointer movingMask )
{
  /** Some typedef's. */
  typedef elx::ElastixMain                            ElastixMainType;
  typedef ElastixMainType::Pointer                    ElastixMainPointer;
  typedef std::vector< ElastixMainPointer >           ElastixMainVectorType;
  typedef ElastixMainType::ObjectPointer              ObjectPointer;
  typedef ElastixMainType::DataObjectContainerType    DataObjectContainerType;
  typedef ElastixMainType::DataObjectContainerPointer DataObjectContainerPointer;
  typedef ElastixMainType::FlatDirectionCosinesType   FlatDirectionCosinesType;

  typedef ElastixMainType::ArgumentMapType ArgumentMapType;
  typedef ArgumentMapType::value_type      ArgumentMapEntryType;

  typedef std::pair< std::string, std::string > ArgPairType;
  typedef std::queue< ArgPairType >             ParameterFileListType;
  typedef ParameterFileListType::value_type     ParameterFileListEntryType;

  // Clear output transform parameters
  this->m_TransformParametersList.clear();

  tmr::Timer::Pointer timer;

  /** Some declarations and initialisations. */
  ElastixMainVectorType elastices;

  ObjectPointer              transform            = 0;
  DataObjectContainerPointer fixedImageContainer  = 0;
  DataObjectContainerPointer movingImageContainer = 0;
  DataObjectContainerPointer fixedMaskContainer   = 0;
  DataObjectContainerPointer movingMaskContainer  = 0;
  DataObjectContainerPointer resultImageContainer = 0;
  FlatDirectionCosinesType   fixedImageOriginalDirection;
  int                        returndummy = 0;
  ArgumentMapType            argMap;
  ParameterFileListType      parameterFileList;
  std::string                outFolder   = "";
  std::string                logFileName = "";
  unsigned short             i;
  std::string                key;
  std::string                value;
  unsigned long              nrOfParameterFiles = parameterMaps.size();

  /** Setup the argumentMap for output path. */
  if( !outputPath.empty() )
  {
    /** Put command line parameters into parameterFileList. */
    key   = "-out";
    value = outputPath;

    /** Make sure that last character of the output folder equals a '/'. */
    if( value.find_last_of( "/" ) != value.size() - 1 )
    {
      value.append( "/" );
    }
  }
  else
  {
    /** Put command line parameters into parameterFileList. */
    //there must be an "-out", this is checked later in code!!
    key   = "-out";
    value = "output_path_not_set";
  }

  /** Save this information. */
  outFolder = value;

  /** Attempt to save the arguments in the ArgumentMap. */
  if( argMap.count( key.c_str() ) == 0 )
  {
    argMap.insert( ArgumentMapEntryType( key.c_str(), value.c_str() ) );
  }
  else if( performCout )
  {
    /** Duplicate arguments. */
    std::cerr << "WARNING!" << std::endl;
    std::cerr << "Argument " << key.c_str() << "is only required once." << std::endl;
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
      return -2;
    }
    else
    {
      /** Setup xout. */
      if( performLogging )
      {
        logFileName = outFolder + "elastix.log";
      }
    }
  }

  /** The argv0 argument, required for finding the component.dll/so's. */
  argMap.insert( ArgumentMapEntryType( "-argv0", "elastix" ) );

  /** Setup xout. */
  returndummy = elx::xoutSetup( logFileName.c_str(), performLogging, performCout );
  if( returndummy && performCout )
  {
    if( performCout )
    {
      std::cerr << "ERROR while setting up xout." << std::endl;
    }
    return returndummy;
  }
  elxout << std::endl;

  /** Declare a timer, start it and print the start time. */
  tmr::Timer::Pointer totaltimer = tmr::Timer::New();
  totaltimer->StartTimer();
  elxout << "elastix is started at " << totaltimer->PrintStartTime()
         << ".\n" << std::endl;

  /************************************************************************
   *                                              *
   *  Generate containers with input images       *
   *                                              *
   ************************************************************************/

  /* Allocate and store images in containers */
  fixedImageContainer                        = DataObjectContainerType::New();
  movingImageContainer                       = DataObjectContainerType::New();
  fixedImageContainer->CreateElementAt( 0 )  = fixedImage;
  movingImageContainer->CreateElementAt( 0 ) = movingImage;

  /* Allocate and store masks in containers if available*/
  if( fixedMask )
  {
    fixedMaskContainer                       = DataObjectContainerType::New();
    fixedMaskContainer->CreateElementAt( 0 ) = fixedMask;
  }
  if( movingMask )
  {
    movingMaskContainer                       = DataObjectContainerType::New();
    movingMaskContainer->CreateElementAt( 0 ) = movingMask;
  }

  //todo original direction cosin, problem is that Image type is unknown at this in elastixlib.cxx
  //for now in elaxElastixTemplate (Run()) direction cosines are taken from fixed image

  /************************************************************************
   *                                                  *
   *    START REGISTRATION                            *
   *  Do the (possibly multiple) registration(s).     *
   *                                                  *
   ************************************************************************/

  for( i = 0; i < nrOfParameterFiles; i++ )
  {
    /** Create another instance of ElastixMain. */
    elastices.push_back( ElastixMainType::New() );

    /** Set stuff we get from a former registration. */
    elastices[ i ]->SetInitialTransform( transform );
    elastices[ i ]->SetFixedImageContainer( fixedImageContainer );
    elastices[ i ]->SetMovingImageContainer( movingImageContainer );
    elastices[ i ]->SetFixedMaskContainer( fixedMaskContainer );
    elastices[ i ]->SetMovingMaskContainer( movingMaskContainer );
    elastices[ i ]->SetResultImageContainer( resultImageContainer );
    elastices[ i ]->SetOriginalFixedImageDirectionFlat( fixedImageOriginalDirection );

    /** Set the current elastix-level. */
    elastices[ i ]->SetElastixLevel( i );
    elastices[ i ]->SetTotalNumberOfElastixLevels( nrOfParameterFiles );

    /** Delete the previous ParameterFileName. */
    if( argMap.count( "-p" ) )
    {
      argMap.erase( "-p" );
    }

    /** Print a start message. */
    elxout << "-------------------------------------------------------------------------" << "\n" << std::endl;
    elxout << "Running elastix with parameter map " << i << std::endl;

    /** Declare a timer, start it and print the start time. */
    timer = tmr::Timer::New();
    timer->StartTimer();
    elxout << "Current time: " << timer->PrintStartTime() << "." << std::endl;

    /** Start registration. */
    returndummy = elastices[ i ]->Run( argMap, parameterMaps[ i ] );

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
    resultImageContainer        = elastices[ i ]->GetResultImageContainer();
    fixedImageOriginalDirection = elastices[ i ]->GetOriginalFixedImageDirectionFlat();

    /** Stop timer and print it. */
    timer->StopTimer();
    elxout << "\nCurrent time: " << timer->PrintStopTime() << "." << std::endl;
    elxout << "Time used for running elastix with this parameter file: "
           << timer->PrintElapsedTimeDHMS() << ".\n" << std::endl;

    /** Get the transformation parameter map. */
    this->m_TransformParametersList.push_back( elastices[ i ]->GetTransformParametersMap() );

    /** Set initial transform to an index number instead of a parameter filename. */
    if( i > 0 )
    {
      std::stringstream toString;
      toString << ( i - 1 );
      this->m_TransformParametersList[ i ][ "InitialTransformParametersFileName" ][ 0 ]
        = toString.str();
    }

    /** Try to release some memory. */
    elastices[ i ] = 0;

  } // end loop over registrations

  elxout << "-------------------------------------------------------------------------"
         << "\n" << std::endl;

  /** Stop totaltimer and print it. */
  totaltimer->StopTimer();
  elxout << "Total time elapsed: " << totaltimer->PrintElapsedTimeDHMS()
         << ".\n" << std::endl;

  /************************************************************************
   *                                *
   *  Cleanup everything            *
   *                                *
   ************************************************************************/

  /*
   *  Make sure all the components that are defined in a Module (.DLL/.so)
   *  are deleted before the modules are closed.
   */
  for( i = 0; i < nrOfParameterFiles; i++ )
  {
    elastices[ i ] = 0;
  }

  /* Set result image for output */
  if( resultImageContainer.IsNotNull() && resultImageContainer->Size() > 0 )
  {
    this->m_ResultImage = resultImageContainer->ElementAt( 0 );
  }

  transform            = 0;
  fixedImageContainer  = 0;
  movingImageContainer = 0;
  fixedMaskContainer   = 0;
  movingMaskContainer  = 0;
  resultImageContainer = 0;

  /** Close the modules. */
  ElastixMainType::UnloadComponents();

  /** Exit and return the error code. */
  return 0;

} // end RegisterImages()


} // end namespace elastix

#endif // end #ifndef __elastixlib_cxx
