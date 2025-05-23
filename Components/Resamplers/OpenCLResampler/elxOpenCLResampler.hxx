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
#ifndef elxOpenCLResampler_hxx
#define elxOpenCLResampler_hxx

#include "elxOpenCLResampler.h"
#include "itkOpenCLLogger.h"

namespace elastix
{

/**
 * ******************* Constructor ***********************
 */

template <typename TElastix>
elastix::OpenCLResampler<TElastix>::OpenCLResampler()
{
  // Check if the OpenCL context has been created.
  itk::OpenCLContext::Pointer context = itk::OpenCLContext::GetInstance();
  this->m_ContextCreated = context->IsCreated();
  if (this->m_ContextCreated)
  {
    this->m_TransformCopier = TransformCopierType::New();
    this->m_InterpolatorCopier = InterpolateCopierType::New();

    try
    {
      this->m_GPUResampler = GPUResamplerType::New();
      this->m_GPUResamplerCreated = true;
    }
    catch (itk::OpenCLCompileError & e)
    {
      // First log then report OpenCL compile error
      itk::OpenCLLogger::Pointer logger = itk::OpenCLLogger::GetInstance();
      logger->Write(itk::LoggerBase::PriorityLevelEnum::CRITICAL, e.GetDescription());

      log::error(std::ostringstream{} << "ERROR: OpenCL program has not been compiled during GPU resampler creation.\n"
                                      << "  Please check the '" << logger->GetLogFileName()
                                      << "' in output directory.");

      this->SwitchingToCPUAndReport(true);
      this->m_GPUResamplerCreated = false;
    }
    catch (itk::ExceptionObject & e)
    {
      log::error(std::ostringstream{} << "ERROR: Exception during GPU resampler creation: " << e);
      this->SwitchingToCPUAndReport(true);
      this->m_GPUResamplerCreated = false;
    }
  }
  else
  {
    this->SwitchingToCPUAndReport(false);
  }

  this->m_UseOpenCL = true;

} // end Constructor


/**
 * ******************* SetTransform ***********************
 */

template <typename TElastix>
void
OpenCLResampler<TElastix>::SetTransform(const TransformType * _arg)
{
  Superclass1::SetTransform(_arg);

  if (this->m_ContextCreated && this->m_GPUResamplerCreated)
  {
    // Cast to the AdvancedCombinationTransform
    const auto * advancedCombinationTransform = dynamic_cast<const AdvancedCombinationTransformType *>(_arg);

    // Set input for the transform copier
    this->m_TransformCopier->SetInputTransform(advancedCombinationTransform);
  }
} // end SetTransform()


/**
 * ******************* SetInterpolator ***********************
 */

template <typename TElastix>
void
OpenCLResampler<TElastix>::SetInterpolator(InterpolatorType * _arg)
{
  if (this->m_ContextCreated && this->m_GPUResamplerCreated)
  {
    // Set input for the interpolate copier
    this->m_InterpolatorCopier->SetInputInterpolator(_arg);
  }
  else
  {
    // Change the default LinearInterpolator only when the OpenCL
    // is not properly setup
    Superclass1::SetInterpolator(_arg);
  }
} // end SetInterpolator()


/**
 * ******************* BeforeGenerateData ***********************
 */

template <typename TElastix>
void
OpenCLResampler<TElastix>::BeforeGenerateData()
{
  // Set it to true, if something goes wrong during configuration, it will be false
  this->m_GPUResamplerReady = true;

  // Local GPU transform, GPU interpolator and GPU input image
  GPUTransformPointer            gpuTransform;
  GPUExplicitInterpolatorPointer gpuInterpolator;
  GPUInputImagePointer           gpuInputImage;

  // Perform transform copy
  try
  {
    this->m_TransformCopier->Update();
    gpuTransform = this->m_TransformCopier->GetModifiableOutput();
  }
  catch (itk::ExceptionObject & e)
  {
    log::error(std::ostringstream{} << "ERROR: Exception during making GPU copy of the transform: " << e);
    this->SwitchingToCPUAndReport(true);
  }

  if (this->m_GPUResamplerReady)
  {
    // Perform interpolator copy
    try
    {
      this->m_InterpolatorCopier->Update();
      gpuInterpolator = this->m_InterpolatorCopier->GetModifiableExplicitOutput();
    }
    catch (itk::ExceptionObject & e)
    {
      log::error(std::ostringstream{} << "ERROR: Exception during making GPU copy of the interpolator: " << e);
      this->SwitchingToCPUAndReport(true);
    }
  }

  if (this->m_GPUResamplerReady)
  {
    // Create GPU input image
    try
    {
      gpuInputImage = GPUInputImageType::New();
      gpuInputImage->GraftITKImage(this->GetInput());
      gpuInputImage->AllocateGPU();
      gpuInputImage->GetGPUDataManager()->SetCPUBufferLock(true);
      gpuInputImage->GetGPUDataManager()->SetGPUDirtyFlag(true);
      gpuInputImage->GetGPUDataManager()->UpdateGPUBuffer();
    }
    catch (itk::ExceptionObject & e)
    {
      log::error(std::ostringstream{} << "ERROR: Exception during creating GPU input image: " << e);
      this->SwitchingToCPUAndReport(true);
    }
  }

  if (this->m_GPUResamplerReady)
  {
    // Set the m_GPUResampler properties the same way as Superclass1
    this->m_GPUResampler->SetSize(this->GetSize());
    this->m_GPUResampler->SetDefaultPixelValue(this->GetDefaultPixelValue());
    this->m_GPUResampler->SetOutputSpacing(this->GetOutputSpacing());
    this->m_GPUResampler->SetOutputOrigin(this->GetOutputOrigin());
    this->m_GPUResampler->SetOutputDirection(this->GetOutputDirection());
    this->m_GPUResampler->SetOutputStartIndex(this->GetOutputStartIndex());
  }

  if (this->m_GPUResamplerReady)
  {
    try
    {
      this->m_GPUResampler->SetInput(gpuInputImage);
      this->m_GPUResampler->SetTransform(gpuTransform);
      this->m_GPUResampler->SetInterpolator(gpuInterpolator);
    }
    catch (itk::OpenCLCompileError & e)
    {
      // First log then report OpenCL compile error
      itk::OpenCLLogger::Pointer logger = itk::OpenCLLogger::GetInstance();
      logger->Write(itk::LoggerBase::PriorityLevelEnum::CRITICAL, e.GetDescription());

      log::error(std::ostringstream{} << "ERROR: OpenCL program has not been compiled during setting GPU resampler.\n"
                                      << "  Please check the '" << logger->GetLogFileName()
                                      << "' in output directory.");

      this->SwitchingToCPUAndReport(true);
    }
    catch (itk::ExceptionObject & e)
    {
      log::error(std::ostringstream{} << "ERROR: Exception during setting GPU resampler: " << e);
      this->SwitchingToCPUAndReport(true);
    }
  }
} // end BeforeGenerateData()


/**
 * ******************* GenerateData ***********************
 */

template <typename TElastix>
void
OpenCLResampler<TElastix>::GenerateData()
{
  if (!this->m_ContextCreated || !this->m_GPUResamplerCreated || !this->m_UseOpenCL)
  {
    // Switch to CPU version
    Superclass1::GenerateData();
    return;
  }

  // First execute BeforeGenerateData to configure GPU resampler
  this->BeforeGenerateData();
  if (!this->m_GPUResamplerReady)
  {
    Superclass1::GenerateData();
    return;
  }

  // Allocate memory
  this->AllocateOutputs();

  // Perform GPU resampler execution
  this->m_GPUResampler->Update();

  // Perform GPU explicit sync and graft the output to this filter
  // itk::GPUExplicitSync< GPUResamplerType, GPUOutputImageType >( this->m_GPUResampler, false );
  this->GraftOutput(this->m_GPUResampler->GetOutput());

  // Report OpenCL device to the log
  this->ReportToLog();
} // end GenerateData()


/**
 * ******************* BeforeRegistration ***********************
 */

template <typename TElastix>
void
OpenCLResampler<TElastix>::BeforeRegistration()
{
  // Are we using a OpenCL enabled GPU for resampling?
  this->m_UseOpenCL = true;
  this->m_Configuration->ReadParameter(this->m_UseOpenCL, "OpenCLResamplerUseOpenCL", 0, false);

} // end BeforeRegistration()


/*
 * ******************* ReadFromFile  ****************************
 */

template <typename TElastix>
void
OpenCLResampler<TElastix>::ReadFromFile()
{
  // Call ReadFromFile of the ResamplerBase.
  this->Superclass2::ReadFromFile();

  // OpenCL resampler specific.
  this->m_UseOpenCL = true;
  this->m_Configuration->ReadParameter(this->m_UseOpenCL, "OpenCLResamplerUseOpenCL", 0);

} // end ReadFromFile()


/**
 * ************************* CreateDerivedTransformParameterMap ************************
 */

template <typename TElastix>
auto
OpenCLResampler<TElastix>::CreateDerivedTransformParameterMap() const -> ParameterMapType
{
  return { { "OpenCLResamplerUseOpenCL", { Conversion::ToString(this->m_UseOpenCL) } } };

} // end CreateDerivedTransformParameterMap()


/**
 * ************************* SwitchingToCPUAndReport ************************
 */

template <typename TElastix>
void
OpenCLResampler<TElastix>::SwitchingToCPUAndReport(const bool configError)
{
  if (!configError)
  {
    log::warn(std::ostringstream{} << "WARNING: The OpenCL context could not be created.\n"
                                   << "  The OpenCLResampler is switching back to CPU mode.");
  }
  else
  {
    log::warn(std::ostringstream{} << "WARNING: Unable to configure the GPU.\n"
                                   << "  The OpenCLResampler is switching back to CPU mode.");
  }
  this->m_GPUResamplerReady = false;

} // end SwitchingToCPUAndReport()


/**
 * ************************* ReportToLog ************************************
 */

template <typename TElastix>
void
OpenCLResampler<TElastix>::ReportToLog()
{
  itk::OpenCLContext::Pointer context = itk::OpenCLContext::GetInstance();
  itk::OpenCLDevice           device = context->GetDefaultDevice();
  log::info(std::ostringstream{} << "  Applying final transform was performed by " << device.GetName() << " from "
                                 << device.GetVendor() << ".");
} // end ReportToLog()


} // end namespace elastix

#endif // end #ifndef elxOpenCLResampler_hxx
