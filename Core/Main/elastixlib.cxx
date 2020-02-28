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

 // Elastix header files:
#include "elastixlib.h"
#include "elxElastixMain.h"
#include "elastix.h" // For ConvertSecondsToDHMS and GetCurrentDateAndTime.

#ifdef _ELASTIX_USE_MEVISDICOMTIFF
#include "itkUseMevisDicomTiff.h"
#endif

// ITK header files:
#include <itkDataObject.h>
#include <itkObject.h>
#include <itkTimeProbe.h>
#include <itksys/SystemInformation.hxx>
#include <itksys/SystemTools.hxx>

// Standard C++ header files:
#include <iostream>
#include <string>
#include <queue>
#include <vector>

namespace elastix
{

/**
 * ******************* Constructor ***********************
 */

ELASTIX::ELASTIX() : m_ResultImage(nullptr)
{
} // end Constructor


/**
 * ******************* Destructor ***********************
 */

ELASTIX::~ELASTIX()
{
  this->m_ResultImage = nullptr;
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
  const ParameterMapType & parameterMap,
  const std::string & outputPath,
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
  const std::vector< ParameterMapType > & parameterMaps,
  const std::string & outputPath,
  bool performLogging,
  bool performCout,
  ImagePointer fixedMask,
  ImagePointer movingMask,
  ObjectPointer transform)
{
  /** Some typedef's. */
  typedef elx::ElastixMain                            ElastixMainType;
  typedef ElastixMainType::Pointer                    ElastixMainPointer;
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

  /** Some declarations and initialisations. */

  //ObjectPointer              transform = 0;
  DataObjectContainerPointer fixedImageContainer  = nullptr;
  DataObjectContainerPointer movingImageContainer = nullptr;
  DataObjectContainerPointer fixedMaskContainer   = nullptr;
  DataObjectContainerPointer movingMaskContainer  = nullptr;
  DataObjectContainerPointer resultImageContainer = nullptr;
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
  itk::TimeProbe totaltimer;
  totaltimer.Start();
  elxout << "elastix is started at " << GetCurrentDateAndTime() << ".\n" << std::endl;

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
    const auto elastixMain = ElastixMainType::New();

    /** Set stuff we get from a former registration. */
    elastixMain->SetInitialTransform( transform );
    elastixMain->SetFixedImageContainer( fixedImageContainer );
    elastixMain->SetMovingImageContainer( movingImageContainer );
    elastixMain->SetFixedMaskContainer( fixedMaskContainer );
    elastixMain->SetMovingMaskContainer( movingMaskContainer );
    elastixMain->SetResultImageContainer( resultImageContainer );
    elastixMain->SetOriginalFixedImageDirectionFlat( fixedImageOriginalDirection );

    /** Set the current elastix-level. */
    elastixMain->SetElastixLevel( i );
    elastixMain->SetTotalNumberOfElastixLevels( nrOfParameterFiles );

    /** Delete the previous ParameterFileName. */
    if( argMap.count( "-p" ) )
    {
      argMap.erase( "-p" );
    }

    /** Print a start message. */
    elxout << "-------------------------------------------------------------------------" << "\n" << std::endl;
    elxout << "Running elastix with parameter map " << i << std::endl;

    /** Declare a timer, start it and print the start time. */
    itk::TimeProbe timer;
    timer.Start();
    elxout << "Current time: " << GetCurrentDateAndTime() << "." << std::endl;

    /** Start registration. */
    returndummy = elastixMain->Run( argMap, parameterMaps[ i ] );

    /** Check for errors. */
    if( returndummy != 0 )
    {
      xl::xout[ "error" ] << "Errors occurred!" << std::endl;
      return returndummy;
    }

    /** Get the transform, the fixedImage and the movingImage
     * in order to put it in the (possibly) next registration.
     */
    transform                   = elastixMain->GetFinalTransform();
    fixedImageContainer         = elastixMain->GetFixedImageContainer();
    movingImageContainer        = elastixMain->GetMovingImageContainer();
    fixedMaskContainer          = elastixMain->GetFixedMaskContainer();
    movingMaskContainer         = elastixMain->GetMovingMaskContainer();
    resultImageContainer        = elastixMain->GetResultImageContainer();
    fixedImageOriginalDirection = elastixMain->GetOriginalFixedImageDirectionFlat();

    /** Stop timer and print it. */
    timer.Stop();
    elxout << "\nCurrent time: " << GetCurrentDateAndTime() << "." << std::endl;
    elxout << "Time used for running elastix with this parameter file: "
           << ConvertSecondsToDHMS( timer.GetMean(), 1 ) << ".\n" << std::endl;

    /** Get the transformation parameter map. */
    this->m_TransformParametersList.push_back( elastixMain->GetTransformParametersMap() );

    /** Set initial transform to an index number instead of a parameter filename. */
    if( i > 0 )
    {
      std::stringstream toString;
      toString << ( i - 1 );
      this->m_TransformParametersList[ i ][ "InitialTransformParametersFileName" ][ 0 ]
        = toString.str();
    }
  } // end loop over registrations

  elxout << "-------------------------------------------------------------------------"
         << "\n" << std::endl;

  /** Stop totaltimer and print it. */
  totaltimer.Stop();
  elxout << "Total time elapsed: "
         << ConvertSecondsToDHMS( totaltimer.GetMean(), 1 ) << ".\n" << std::endl;

  /************************************************************************
   *                                *
   *  Cleanup everything            *
   *                                *
   ************************************************************************/

  /*
   *  Make sure all the components that are defined in a Module (.DLL/.so)
   *  are deleted before the modules are closed.
   */

  /* Set result image for output */
  if( resultImageContainer.IsNotNull() && resultImageContainer->Size() > 0 && resultImageContainer->ElementAt( 0 ).IsNotNull() )
  {
    this->m_ResultImage = resultImageContainer->ElementAt( 0 );
  }

  transform            = nullptr;
  fixedImageContainer  = nullptr;
  movingImageContainer = nullptr;
  fixedMaskContainer   = nullptr;
  movingMaskContainer  = nullptr;
  resultImageContainer = nullptr;

  /** Close the modules. */
  ElastixMainType::UnloadComponents();

  /** Exit and return the error code. */
  return 0;

} // end RegisterImages()


} // end namespace elastix
