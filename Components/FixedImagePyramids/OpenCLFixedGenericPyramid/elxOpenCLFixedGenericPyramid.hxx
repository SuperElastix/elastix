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
#ifndef elxOpenCLFixedGenericPyramid_hxx
#define elxOpenCLFixedGenericPyramid_hxx

#include "elxOpenCLSupportedImageTypes.h"
#include "elxOpenCLFixedGenericPyramid.h"

// GPU includes
#include "itkGPUImageFactory.h"
#include "itkOpenCLLogger.h"

// GPU factory includes
#include "itkGPURecursiveGaussianImageFilterFactory.h"
#include "itkGPUCastImageFilterFactory.h"
#include "itkGPUShrinkImageFilterFactory.h"
#include "itkGPUResampleImageFilterFactory.h"
#include "itkGPUIdentityTransformFactory.h"
#include "itkGPULinearInterpolateImageFunctionFactory.h"

namespace elastix
{

/**
 * ******************* Constructor ***********************
 */

template <class TElastix>
OpenCLFixedGenericPyramid<TElastix>::OpenCLFixedGenericPyramid()
  : m_GPUPyramidReady(true)
  , m_GPUPyramidCreated(true)
  , m_ContextCreated(false)
  , m_UseOpenCL(true)
{
  // Based on the Insight Journal paper:
  // http://insight-journal.org/browse/publication/884
  // it is not beneficial to create pyramids for 2D images with OpenCL.
  // There are also small extra overhead and potential problems may appear.
  // To avoid it, we simply run it on CPU for 2D images.
  if (ImageDimension <= 2)
  {
    xl::xout["warning"] << "WARNING: Creating the fixed pyramid with OpenCL for 2D images is not beneficial.\n";
    xl::xout["warning"] << "  The OpenCLFixedGenericPyramid is switching back to CPU mode." << std::endl;
    return;
  }

  // Check if the OpenCL context has been created.
  itk::OpenCLContext::Pointer context = itk::OpenCLContext::GetInstance();
  this->m_ContextCreated = context->IsCreated();
  if (this->m_ContextCreated)
  {
    try
    {
      this->m_GPUPyramid = GPUPyramidType::New();
    }
    catch (itk::ExceptionObject & e)
    {
      xl::xout["error"] << "ERROR: Exception during GPU fixed generic pyramid creation: " << e << std::endl;
      this->SwitchingToCPUAndReport(true);
      this->m_GPUPyramidCreated = false;
    }
  }
  else
  {
    this->SwitchingToCPUAndReport(false);
  }
} // end Constructor


/**
 * ******************* BeforeGenerateData ***********************
 */

template <class TElastix>
void
OpenCLFixedGenericPyramid<TElastix>::BeforeGenerateData()
{
  // Local GPU input image
  GPUInputImagePointer gpuInputImage;

  if (this->m_GPUPyramidReady)
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
      xl::xout["error"] << "ERROR: Exception during creating GPU input image for fixed generic pyramid: " << e
                        << std::endl;
      this->SwitchingToCPUAndReport(true);
    }
  }

  if (this->m_GPUPyramidReady)
  {
    // Set the m_GPUResampler properties the same way as Superclass1
    this->m_GPUPyramid->SetNumberOfLevels(this->GetNumberOfLevels());
    this->m_GPUPyramid->SetRescaleSchedule(this->GetRescaleSchedule());
    this->m_GPUPyramid->SetSmoothingSchedule(this->GetSmoothingSchedule());
    this->m_GPUPyramid->SetUseShrinkImageFilter(this->GetUseShrinkImageFilter());
    this->m_GPUPyramid->SetComputeOnlyForCurrentLevel(this->GetComputeOnlyForCurrentLevel());
  }

  if (this->m_GPUPyramidReady)
  {
    try
    {
      this->m_GPUPyramid->SetInput(gpuInputImage);
    }
    catch (itk::ExceptionObject & e)
    {
      xl::xout["error"] << "ERROR: Exception during setting GPU fixed generic pyramid: " << e << std::endl;
      this->SwitchingToCPUAndReport(true);
    }
  }
} // end BeforeGenerateData()


/**
 * ******************* GenerateData ***********************
 */

template <class TElastix>
void
OpenCLFixedGenericPyramid<TElastix>::GenerateData()
{
  if (!this->m_ContextCreated || !this->m_GPUPyramidCreated || !this->m_UseOpenCL || !this->m_GPUPyramidReady)
  {
    // Switch to CPU version
    Superclass1::GenerateData();
    return;
  }

  // First execute BeforeGenerateData to configure GPU pyramid
  this->BeforeGenerateData();
  if (!this->m_GPUPyramidReady)
  {
    Superclass1::GenerateData();
    return;
  }

  bool computedUsingOpenCL = true;

  // Register factories
  this->RegisterFactories();
  try
  {
    // Perform GPU pyramid execution
    this->m_GPUPyramid->Update();
  }
  catch (itk::OpenCLCompileError & e)
  {
    // First log then report OpenCL compile error
    itk::OpenCLLogger::Pointer logger = itk::OpenCLLogger::GetInstance();
    logger->Write(itk::LoggerBase::PriorityLevelEnum::CRITICAL, e.GetDescription());

    xl::xout["error"] << "ERROR: OpenCL program has not been compiled during updating GPU fixed pyramid calculation.\n"
                      << "  Please check the '" << logger->GetLogFileName() << "' in output directory." << std::endl;
    computedUsingOpenCL = false;
  }
  catch (itk::ExceptionObject & e)
  {
    xl::xout["error"] << "ERROR: Exception during updating GPU fixed pyramid calculation: " << e << std::endl;
    computedUsingOpenCL = false;
  }
  catch (...)
  {
    xl::xout["error"] << "ERROR: Unknown exception during updating GPU fixed pyramid calculation." << std::endl;
    computedUsingOpenCL = false;
  }

  // Unregister factories
  this->UnregisterFactories();

  if (computedUsingOpenCL)
  {
    // Graft outputs
    const auto numberOfLevels = this->GetNumberOfLevels();
    for (unsigned int i = 0; i < numberOfLevels; i++)
    {
      this->GraftNthOutput(i, this->m_GPUPyramid->GetOutput(i));
    }

    // Report OpenCL device to the log
    this->ReportToLog();
  }
  else
  {
    xl::xout["warning"] << "WARNING: The fixed pyramid computation with OpenCL failed due to the error.\n";
    xl::xout["warning"] << "  The OpenCLFixedGenericImagePyramid is switching back to CPU mode." << std::endl;
    Superclass1::GenerateData();
  }
} // end GenerateData()


/**
 * ******************* RegisterFactories ***********************
 */

template <class TElastix>
void
OpenCLFixedGenericPyramid<TElastix>::RegisterFactories()
{
  // Typedefs for factories
  using ImageFactoryType = itk::GPUImageFactory2<OpenCLImageTypes, OpenCLImageDimentions>;
  using RecursiveGaussianFactoryType =
    itk::GPURecursiveGaussianImageFilterFactory2<OpenCLImageTypes, OpenCLImageTypes, OpenCLImageDimentions>;
  using CastFactoryType = itk::GPUCastImageFilterFactory2<OpenCLImageTypes, OpenCLImageTypes, OpenCLImageDimentions>;
  using ShrinkFactoryType =
    itk::GPUShrinkImageFilterFactory2<OpenCLImageTypes, OpenCLImageTypes, OpenCLImageDimentions>;
  using ResampleFactoryType =
    itk::GPUResampleImageFilterFactory2<OpenCLImageTypes, OpenCLImageTypes, OpenCLImageDimentions>;
  using IdentityFactoryType = itk::GPUIdentityTransformFactory2<OpenCLImageDimentions>;
  using LinearFactoryType = itk::GPULinearInterpolateImageFunctionFactory2<OpenCLImageTypes, OpenCLImageDimentions>;

  // Create factories
  auto imageFactory = ImageFactoryType::New();
  auto recursiveFactory = RecursiveGaussianFactoryType::New();
  auto castFactory = CastFactoryType::New();
  auto shrinkFactory = ShrinkFactoryType::New();
  auto resampleFactory = ResampleFactoryType::New();
  auto identityFactory = IdentityFactoryType::New();
  auto linearFactory = LinearFactoryType::New();

  // Register factories
  itk::ObjectFactoryBase::RegisterFactory(imageFactory);
  itk::ObjectFactoryBase::RegisterFactory(recursiveFactory);
  itk::ObjectFactoryBase::RegisterFactory(castFactory);
  itk::ObjectFactoryBase::RegisterFactory(shrinkFactory);
  itk::ObjectFactoryBase::RegisterFactory(resampleFactory);
  itk::ObjectFactoryBase::RegisterFactory(identityFactory);
  itk::ObjectFactoryBase::RegisterFactory(linearFactory);

  // Append them
  this->m_Factories.push_back(imageFactory.GetPointer());
  this->m_Factories.push_back(recursiveFactory.GetPointer());
  this->m_Factories.push_back(castFactory.GetPointer());
  this->m_Factories.push_back(shrinkFactory.GetPointer());
  this->m_Factories.push_back(resampleFactory.GetPointer());
  this->m_Factories.push_back(identityFactory.GetPointer());
  this->m_Factories.push_back(linearFactory.GetPointer());

} // end RegisterFactories()


/**
 * ******************* UnregisterFactories ***********************
 */

template <class TElastix>
void
OpenCLFixedGenericPyramid<TElastix>::UnregisterFactories()
{
  for (std::vector<ObjectFactoryBasePointer>::iterator it = this->m_Factories.begin(); it != this->m_Factories.end();
       ++it)
  {
    itk::ObjectFactoryBase::UnRegisterFactory(*it);
  }
  this->m_Factories.clear();
} // end UnregisterFactories()


/**
 * ******************* BeforeRegistration ***********************
 */

template <class TElastix>
void
OpenCLFixedGenericPyramid<TElastix>::BeforeRegistration()
{
  // Are we using a OpenCL enabled GPU for pyramid?
  this->m_UseOpenCL = true;
  this->m_Configuration->ReadParameter(this->m_UseOpenCL, "OpenCLFixedGenericImagePyramidUseOpenCL", 0);

} // end BeforeRegistration()


/*
 * ******************* ReadFromFile  ****************************
 */

template <class TElastix>
void
OpenCLFixedGenericPyramid<TElastix>::ReadFromFile()
{
  // OpenCL pyramid specific.
  this->m_UseOpenCL = true;
  this->m_Configuration->ReadParameter(this->m_UseOpenCL, "OpenCLFixedGenericImagePyramidUseOpenCL", 0);

} // end ReadFromFile()


/**
 * ************************* SwitchingToCPUAndReport ************************
 */

template <class TElastix>
void
OpenCLFixedGenericPyramid<TElastix>::SwitchingToCPUAndReport(const bool configError)
{
  if (!configError)
  {
    xl::xout["warning"] << "WARNING: The OpenCL context could not be created.\n";
    xl::xout["warning"] << "  The OpenCLFixedGenericImagePyramid is switching back to CPU mode." << std::endl;
  }
  else
  {
    xl::xout["warning"] << "WARNING: Unable to configure the GPU.\n";
    xl::xout["warning"] << "  The OpenCLFixedGenericImagePyramid is switching back to CPU mode." << std::endl;
  }
  this->m_GPUPyramidReady = false;

} // end SwitchingToCPUAndReport()


/**
 * ************************* ReportToLog ************************************
 */

template <class TElastix>
void
OpenCLFixedGenericPyramid<TElastix>::ReportToLog()
{
  itk::OpenCLContext::Pointer context = itk::OpenCLContext::GetInstance();
  itk::OpenCLDevice           device = context->GetDefaultDevice();
  elxout << "  Fixed pyramid was computed by " << device.GetName() << " from " << device.GetVendor() << ".";
} // end ReportToLog()


} // end namespace elastix

#endif // end #ifndef elxOpenCLFixedGenericPyramid_hxx
