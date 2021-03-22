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

#if defined(_WIN32) && !defined(__CYGWIN__)
#  include <windows.h>
#endif

#include "elxTransformixMain.h"

#include "elxMacro.h"

#ifdef ELASTIX_USE_OPENCL
#  include "itkOpenCLContext.h"
#  include "itkOpenCLSetup.h"
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
TransformixMain::Run(void)
{
  /** Set process properties. */
  this->SetProcessPriority();
  this->SetMaximumNumberOfThreads();

  /** Initialize database. */
  int errorCode = this->InitDBIndex();
  if (errorCode != 0)
  {
    return errorCode;
  }

  /** Create the Elastix component. */
  try
  {
    /** Key "Elastix", see elxComponentLoader::InstallSupportedImageTypes(). */
    this->m_Elastix = this->CreateComponent("Elastix");
  }
  catch (itk::ExceptionObject & excp)
  {
    /** We just print the exception and let the program quit. */
    xl::xout["error"] << excp << std::endl;
    errorCode = 1;
    return errorCode;
  }

  /** Create OpenCL context and logger here. */
#ifdef ELASTIX_USE_OPENCL
  /** Check if user overrides OpenCL device selection. */
  std::string userSuppliedOpenCLDeviceType = "GPU";
  this->m_Configuration->ReadParameter(userSuppliedOpenCLDeviceType, "OpenCLDeviceType", 0, false);

  int userSuppliedOpenCLDeviceID = -1;
  this->m_Configuration->ReadParameter(userSuppliedOpenCLDeviceID, "OpenCLDeviceID", 0, false);

  std::string errorMessage = "";
  const bool  creatingContextSuccessful =
    itk::CreateOpenCLContext(errorMessage, userSuppliedOpenCLDeviceType, userSuppliedOpenCLDeviceID);
  if (!creatingContextSuccessful)
  {
    /** Report and disable the GPU by releasing the context. */
    elxout << errorMessage << std::endl;
    elxout << "  OpenCL processing in transformix is disabled." << std::endl << std::endl;

    itk::OpenCLContext::Pointer context = itk::OpenCLContext::GetInstance();
    context->Release();
  }

  /** Create a log file. */
  itk::CreateOpenCLLogger("transformix", this->m_Configuration->GetCommandLineArgument("-out"));
#endif
  auto & elastixBase = this->GetElastixBase();

  if (BaseComponent::IsElastixLibrary())
  {
    elastixBase.SetConfigurations(this->m_Configurations);
  }

  /** Set some information in the ElastixBase. */
  elastixBase.SetConfiguration(this->m_Configuration);
  elastixBase.SetDBIndex(this->m_DBIndex);

  /** Populate the component containers. No default is specified for the Transform. */
  elastixBase.SetResampleInterpolatorContainer(
    this->CreateComponents("ResampleInterpolator", "FinalBSplineInterpolator", errorCode));

  elastixBase.SetResamplerContainer(this->CreateComponents("Resampler", "DefaultResampler", errorCode));

  elastixBase.SetTransformContainer(this->CreateComponents("Transform", "", errorCode));

  /** Check if all components could be created. */
  if (errorCode != 0)
  {
    xl::xout["error"] << "ERROR:" << std::endl;
    xl::xout["error"] << "One or more components could not be created." << std::endl;
    return 1;
  }

  /** Set the images. If not set by the user, it is not a problem.
   * ElastixTemplate will try to load them from disk.
   */
  elastixBase.SetMovingImageContainer(this->GetModifiableMovingImageContainer());

  /** Set the initial transform, if it happens to be there
   * \todo: Does this make sense for transformix?
   */
  elastixBase.SetInitialTransform(this->GetModifiableInitialTransform());

  /** ApplyTransform! */
  try
  {
    errorCode = elastixBase.ApplyTransform();
  }
  catch (itk::ExceptionObject & excp)
  {
    /** We just print the exception and let the program quit. */
    xl::xout["error"] << std::endl
                      << "--------------- Exception ---------------" << std::endl
                      << excp << "-----------------------------------------" << std::endl;
    errorCode = 1;
  }

  /** Save the image container. */
  this->SetMovingImageContainer(elastixBase.GetMovingImageContainer());
  this->SetResultImageContainer(elastixBase.GetResultImageContainer());
  this->SetResultDeformationFieldContainer(elastixBase.GetResultDeformationFieldContainer());

  return errorCode;

} // end Run()


/**
 * **************************** Run *****************************
 */

int
TransformixMain::Run(const ArgumentMapType & argmap)
{
  this->EnterCommandLineArguments(argmap);
  return this->Run();
} // end Run()


/**
 * **************************** Run *****************************
 */

int
TransformixMain::Run(const ArgumentMapType & argmap, const ParameterMapType & inputMap)
{
  this->EnterCommandLineArguments(argmap, inputMap);
  return this->Run();
} // end Run()


/**
 * **************************** Run *****************************
 */

int
TransformixMain::Run(const ArgumentMapType & argmap, const std::vector<ParameterMapType> & inputMaps)
{
  this->EnterCommandLineArguments(argmap, inputMaps);
  return this->Run();
} // end Run()


/**
 * ********************* SetInputImage **************************
 */

void
TransformixMain::SetInputImageContainer(DataObjectContainerType * inputImageContainer)
{
  /** InputImage == MovingImage. */
  this->SetMovingImageContainer(inputImageContainer);

} // end SetInputImage()


/**
 * ********************** Destructor ****************************
 */

TransformixMain::~TransformixMain()
{
#ifdef ELASTIX_USE_OPENCL
  itk::OpenCLContext::Pointer context = itk::OpenCLContext::GetInstance();
  if (context->IsCreated())
  {
    context->Release();
  }
#endif
} // end Destructor


/**
 * ********************* InitDBIndex ****************************
 */

int
TransformixMain::InitDBIndex(void)
{
  /** Check if configuration object was already initialized. */
  if (this->m_Configuration->IsInitialized())
  {
    /** Try to read MovingImagePixelType from the parameter file. */
    this->m_MovingImagePixelType = "float"; // \note: this assumes elastix was compiled for float
    this->m_Configuration->ReadParameter(this->m_MovingImagePixelType, "MovingInternalImagePixelType", 0);

    /** Try to read FixedImagePixelType from the parameter file. */
    this->m_FixedImagePixelType = "float"; // \note: this assumes elastix was compiled for float
    this->m_Configuration->ReadParameter(this->m_FixedImagePixelType, "FixedInternalImagePixelType", 0);

    /** MovingImageDimension. */
    if (this->m_MovingImageDimension == 0)
    {
      /** Try to read it from the transform parameter file. */
      this->m_Configuration->ReadParameter(this->m_MovingImageDimension, "MovingImageDimension", 0);

      if (this->m_MovingImageDimension == 0)
      {
        xl::xout["error"] << "ERROR:" << std::endl;
        xl::xout["error"] << "The MovingImageDimension is not given." << std::endl;
        return 1;
      }
    }

    /** FixedImageDimension. */
    if (this->m_FixedImageDimension == 0)
    {
      /** Try to read it from the transform parameter file. */
      this->m_Configuration->ReadParameter(this->m_FixedImageDimension, "FixedImageDimension", 0);

      if (this->m_FixedImageDimension == 0)
      {
        xl::xout["error"] << "ERROR:" << std::endl;
        xl::xout["error"] << "The FixedImageDimension is not given." << std::endl;
        return 1;
      }
    }

    /** Get the DBIndex from the ComponentDatabase. */
    this->m_DBIndex = this->GetComponentDatabase().GetIndex(this->m_FixedImagePixelType,
                                                            this->m_FixedImageDimension,
                                                            this->m_MovingImagePixelType,
                                                            this->m_MovingImageDimension);
    if (this->m_DBIndex == 0)
    {
      xl::xout["error"] << "ERROR:" << std::endl;
      xl::xout["error"] << "Something went wrong in the ComponentDatabase." << std::endl;
      return 1;
    }

  } // end if m_Configuration->Initialized();
  else
  {
    xl::xout["error"] << "ERROR:" << std::endl;
    xl::xout["error"] << "The configuration object has not been initialized." << std::endl;
    return 1;
  }

  /** Everything is OK! */
  return 0;

} // end InitDBIndex()


} // end namespace elastix
