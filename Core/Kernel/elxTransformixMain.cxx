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
#include "elxDeref.h"

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
TransformixMain::Run()
{
  return RunWithTransform(nullptr);
}

/**
 * **************************** RunWithTransform *****************************
 *
 * Assuming EnterCommandLineParameters has already been invoked.
 * or that m_Configuration is initialized in another way.
 */

int
TransformixMain::RunWithTransform(itk::TransformBase * const transform)
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
  catch (const itk::ExceptionObject & excp)
  {
    /** We just print the exception and let the program quit. */
    log::error(std::ostringstream{} << excp);
    errorCode = 1;
    return errorCode;
  }

  const Configuration & configuration = Deref(MainBase::GetConfiguration());

  /** Create OpenCL context and logger here. */
#ifdef ELASTIX_USE_OPENCL
  /** Check if user overrides OpenCL device selection. */
  std::string userSuppliedOpenCLDeviceType = "GPU";
  configuration.ReadParameter(userSuppliedOpenCLDeviceType, "OpenCLDeviceType", 0, false);

  int userSuppliedOpenCLDeviceID = -1;
  configuration.ReadParameter(userSuppliedOpenCLDeviceID, "OpenCLDeviceID", 0, false);

  std::string errorMessage = "";
  const bool  creatingContextSuccessful =
    itk::CreateOpenCLContext(errorMessage, userSuppliedOpenCLDeviceType, userSuppliedOpenCLDeviceID);
  if (!creatingContextSuccessful)
  {
    /** Report and disable the GPU by releasing the context. */
    log::info(std::ostringstream{} << errorMessage << '\n' << "  OpenCL processing in transformix is disabled.\n");

    itk::OpenCLContext::Pointer context = itk::OpenCLContext::GetInstance();
    context->Release();
  }

  /** Create a log file. */
  itk::CreateOpenCLLogger("transformix", configuration.GetCommandLineArgument("-out"));
#endif
  auto & elastixBase = this->GetElastixBase();

  if (BaseComponent::IsElastixLibrary())
  {
    elastixBase.SetConfigurations(this->m_Configurations);
  }

  /** Set some information in the ElastixBase. */
  elastixBase.SetConfiguration(MainBase::GetConfiguration());
  elastixBase.SetDBIndex(this->m_DBIndex);

  /** Populate the component containers. No default is specified for the Transform. */
  elastixBase.SetResampleInterpolatorContainer(
    this->CreateComponents("ResampleInterpolator", "FinalBSplineInterpolator", errorCode));

  elastixBase.SetResamplerContainer(this->CreateComponents("Resampler", "DefaultResampler", errorCode));

  if (transform)
  {
    const auto transformContainer = elx::ElastixBase::ObjectContainerType::New();
    transformContainer->push_back(transform);
    elastixBase.SetTransformContainer(transformContainer);
  }
  else
  {
    elastixBase.SetTransformContainer(this->CreateComponents("Transform", "", errorCode));
  }

  /** Check if all components could be created. */
  if (errorCode != 0)
  {
    log::error("ERROR: One or more components could not be created.");
    return 1;
  }

  /** Set the images. If not set by the user, it is not a problem.
   * ElastixTemplate will try to load them from disk.
   */
  elastixBase.SetMovingImageContainer(this->GetModifiableMovingImageContainer());

  /** ApplyTransform! */
  try
  {
    errorCode = elastixBase.ApplyTransform(transform == nullptr);
  }
  catch (const itk::ExceptionObject & excp)
  {
    /** We just print the exception and let the program quit. */
    log::error(std::ostringstream{} << "Exception while trying to apply a tranformation:\n" << excp);
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
TransformixMain::Run(const ArgumentMapType &               argmap,
                     const std::vector<ParameterMapType> & inputMaps,
                     itk::TransformBase * const            transform)
{
  this->EnterCommandLineArguments(argmap, inputMaps);
  return this->RunWithTransform(transform);
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
TransformixMain::InitDBIndex()
{
  const Configuration & configuration = Deref(MainBase::GetConfiguration());

  /** Check if configuration object was already initialized. */
  if (configuration.IsInitialized())
  {
    /** Try to read MovingImagePixelType from the parameter file. */
    this->m_MovingImagePixelType = "float"; // \note: this assumes elastix was compiled for float
    configuration.ReadParameter(this->m_MovingImagePixelType, "MovingInternalImagePixelType", 0);

    /** Try to read FixedImagePixelType from the parameter file. */
    this->m_FixedImagePixelType = "float"; // \note: this assumes elastix was compiled for float
    configuration.ReadParameter(this->m_FixedImagePixelType, "FixedInternalImagePixelType", 0);

    /** MovingImageDimension. */
    if (this->m_MovingImageDimension == 0)
    {
      /** Try to read it from the transform parameter file. */
      configuration.ReadParameter(this->m_MovingImageDimension, "MovingImageDimension", 0);

      if (this->m_MovingImageDimension == 0)
      {
        log::error("ERROR: The MovingImageDimension is not given.");
        return 1;
      }
    }

    /** FixedImageDimension. */
    if (this->m_FixedImageDimension == 0)
    {
      /** Try to read it from the transform parameter file. */
      configuration.ReadParameter(this->m_FixedImageDimension, "FixedImageDimension", 0);

      if (this->m_FixedImageDimension == 0)
      {
        log::error("ERROR: The FixedImageDimension is not given.");
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
      log::error("ERROR: Something went wrong in the ComponentDatabase.");
      return 1;
    }

  } // end if configuration.Initialized();
  else
  {
    log::error("ERROR: The configuration object has not been initialized.");
    return 1;
  }

  /** Everything is OK! */
  return 0;

} // end InitDBIndex()


} // end namespace elastix
