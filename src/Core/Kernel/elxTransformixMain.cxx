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

/** If running on a Windows-system, include "windows.h".
 *  This is to set the priority, but which does not work on cygwin.
 */

#if defined( _WIN32 ) && !defined( __CYGWIN__ )
  #include <windows.h>
#endif

#include "elxTransformixMain.h"

#include "elxMacro.h"

#ifdef ELASTIX_USE_OPENCL
#include "itkOpenCLSetup.h"
#endif

namespace elastix
{

/**
 * **************************** Run *****************************
 *
 * Assuming EnterCommandLineParameters has already been invoked.
 * or that m_Configuration is initialized in another way.
 */

int
TransformixMain::Run( void )
{
  /** Set process properties. */
  this->SetProcessPriority();
  this->SetMaximumNumberOfThreads();

  /** Initialize database. */
  int errorCode = this->InitDBIndex();
  if( errorCode != 0 )
  {
    return errorCode;
  }

  /** Create the Elastix component. */
  try
  {
    /** Key "Elastix", see elxComponentLoader::InstallSupportedImageTypes(). */
    this->m_Elastix = this->CreateComponent( "Elastix" );
  }
  catch( itk::ExceptionObject & excp )
  {
    /** We just print the exception and let the program quit. */
    xl::xout[ "error" ] << excp << std::endl;
    errorCode = 1;
    return errorCode;
  }

  /** Create OpenCL context and logger here. */
#ifdef ELASTIX_USE_OPENCL
  std::string errorMessage = "";
  bool errorCreatingContext = itk::CreateOpenCLContext( errorMessage );
  if( errorCreatingContext ){ elxout << errorMessage << std::endl; }
  itk::CreateOpenCLLogger( "transformix", this->m_Configuration->GetCommandLineArgument( "-out" ) );
#endif

#ifdef _ELASTIX_BUILD_LIBRARY
  this->GetElastixBase()->SetConfigurations( this->m_Configurations );
#endif // #ifdef _ELASTIX_BUILD_LIBRARY

  /** Set some information in the ElastixBase. */
  this->GetElastixBase()->SetConfiguration( this->m_Configuration );
  this->GetElastixBase()->SetComponentDatabase( this->s_CDB );
  this->GetElastixBase()->SetDBIndex( this->m_DBIndex );

  /** Populate the component containers. No default is specified for the Transform. */
  this->GetElastixBase()->SetResampleInterpolatorContainer(
    this->CreateComponents( "ResampleInterpolator", "FinalBSplineInterpolator",
    errorCode ) );

  this->GetElastixBase()->SetResamplerContainer(
    this->CreateComponents( "Resampler", "DefaultResampler", errorCode ) );

  this->GetElastixBase()->SetTransformContainer(
    this->CreateComponents( "Transform", "", errorCode ) );

  /** Check if all components could be created. */
  if( errorCode != 0 )
  {
    xl::xout[ "error" ] << "ERROR:" << std::endl;
    xl::xout[ "error" ] << "One or more components could not be created." << std::endl;
    return 1;
  }

  /** Set the images. If not set by the user, it is not a problem.
   * ElastixTemplate will try to load them from disk.
   */
  this->GetElastixBase()->SetMovingImageContainer(
    this->GetMovingImageContainer() );

  /** Set the initial transform, if it happens to be there
  * \todo: Does this make sense for transformix?
  */
  this->GetElastixBase()->SetInitialTransform( this->GetInitialTransform() );

  /** ApplyTransform! */
  try
  {
    errorCode = this->GetElastixBase()->ApplyTransform();
  }
  catch( itk::ExceptionObject & excp )
  {
    /** We just print the exception and let the program quit. */
    xl::xout[ "error" ] << std::endl
                        << "--------------- Exception ---------------"
                        << std::endl << excp
                        << "-----------------------------------------" << std::endl;
    errorCode = 1;
  }

  /** Save the image container. */
  this->SetMovingImageContainer(
    this->GetElastixBase()->GetMovingImageContainer() );
  this->SetResultImageContainer(
    this->GetElastixBase()->GetResultImageContainer() );

  return errorCode;

} // end Run()


/**
 * **************************** Run *****************************
 */

int
TransformixMain::Run( ArgumentMapType & argmap )
{
  this->EnterCommandLineArguments( argmap );
  return this->Run();
} // end Run()


/**
 * **************************** Run *****************************
 */

int
TransformixMain::Run(
  ArgumentMapType & argmap,
  ParameterMapType & inputMap )
{
  this->EnterCommandLineArguments( argmap, inputMap );
  return this->Run();
} // end Run()


/**
 * **************************** Run *****************************
 */

int
TransformixMain::Run(
  ArgumentMapType & argmap,
  std::vector< ParameterMapType > & inputMaps )
{
  this->EnterCommandLineArguments( argmap, inputMaps );
  return this->Run();
} // end Run()


/**
 * ********************* SetInputImage **************************
 */

void
TransformixMain::SetInputImageContainer(
  DataObjectContainerType * inputImageContainer )
{
  /** InputImage == MovingImage. */
  this->SetMovingImageContainer( inputImageContainer );

} // end SetInputImage()


/**
 * ********************* InitDBIndex ****************************
 */

int
TransformixMain::InitDBIndex( void )
{
  /** Check if configuration object was already initialized. */
  if( this->m_Configuration->IsInitialized() )
  {
    /** Try to read MovingImagePixelType from the parameter file. */
    this->m_MovingImagePixelType = "float"; // \note: this assumes elastix was compiled for float
    this->m_Configuration->ReadParameter( this->m_MovingImagePixelType,
      "MovingInternalImagePixelType", 0 );

    /** Try to read FixedImagePixelType from the parameter file. */
    this->m_FixedImagePixelType = "float"; // \note: this assumes elastix was compiled for float
    this->m_Configuration->ReadParameter( this->m_FixedImagePixelType,
      "FixedInternalImagePixelType", 0 );

    /** MovingImageDimension. */
    if( this->m_MovingImageDimension == 0 )
    {
      /** Try to read it from the transform parameter file. */
      this->m_Configuration->ReadParameter( this->m_MovingImageDimension,
        "MovingImageDimension", 0 );

      if( this->m_MovingImageDimension == 0 )
      {
        xl::xout[ "error" ] << "ERROR:" << std::endl;
        xl::xout[ "error" ] << "The MovingImageDimension is not given." << std::endl;
        return 1;
      }
    }

    /** FixedImageDimension. */
    if( this->m_FixedImageDimension == 0 )
    {
      /** Try to read it from the transform parameter file. */
      this->m_Configuration->ReadParameter( this->m_FixedImageDimension,
        "FixedImageDimension", 0 );

      if( this->m_FixedImageDimension == 0 )
      {
        xl::xout[ "error" ] << "ERROR:" << std::endl;
        xl::xout[ "error" ] << "The FixedImageDimension is not given." << std::endl;
        return 1;
      }
    }

    /** Load the components. */
    if( this->s_CDB.IsNull() )
    {
      int loadReturnCode = this->LoadComponents();
      if( loadReturnCode != 0 )
      {
        xl::xout[ "error" ] << "Loading components failed" << std::endl;
        return loadReturnCode;
      }
    }

    if( this->s_CDB.IsNotNull() )
    {
      /** Get the DBIndex from the ComponentDatabase. */
      this->m_DBIndex = this->s_CDB->GetIndex(
        this->m_FixedImagePixelType,
        this->m_FixedImageDimension,
        this->m_MovingImagePixelType,
        this->m_MovingImageDimension );
      if( this->m_DBIndex == 0 )
      {
        xl::xout[ "error" ] << "ERROR:" << std::endl;
        xl::xout[ "error" ] << "Something went wrong in the ComponentDatabase." << std::endl;
        return 1;
      }
    } //end if s_CDB!=0

  } // end if m_Configuration->Initialized();
  else
  {
    xl::xout[ "error" ] << "ERROR:" << std::endl;
    xl::xout[ "error" ] << "The configuration object has not been initialized." << std::endl;
    return 1;
  }

  /** Everything is OK! */
  return 0;

} // end InitDBIndex()


} // end namespace elastix
