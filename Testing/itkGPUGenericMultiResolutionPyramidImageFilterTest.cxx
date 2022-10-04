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
#include "itkTestHelper.h"
#include "itkGenericMultiResolutionPyramidImageFilter.h"

// GPU includes
#include "itkGPUImageFactory.h"
#include "itkGPURecursiveGaussianImageFilterFactory.h"
#include "itkGPUCastImageFilterFactory.h"
#include "itkGPUShrinkImageFilterFactory.h"
#include "itkGPUResampleImageFilterFactory.h"
#include "itkGPUIdentityTransformFactory.h"
#include "itkGPULinearInterpolateImageFunctionFactory.h"

#include "itkOpenCLContextScopeGuard.h"

// ITK include files
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionConstIterator.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"
#include "itkTimeProbe.h"

#include <iomanip> // setprecision, etc.
#include <random>  // For mt19937.

//------------------------------------------------------------------------------
// This test compares the CPU with the GPU version of the
// GenericMultiResolutionPyramidImageFilter.
// The filter takes an input image and produces an output image.
// We compare the CPU and GPU output image write RMSE and speed.

template <typename FilterType>
void
UpdateFilterNTimes(typename FilterType::Pointer filter, const unsigned int N, const bool computeOnlyForCurrentLevel)
{
  for (unsigned int i = 0; i < N; ++i)
  {
    filter->Modified();

    if (!computeOnlyForCurrentLevel)
    {
      filter->Update();
    }
    else
    {
      for (unsigned int j = 0; j < filter->GetNumberOfLevels(); ++j)
      {
        filter->SetCurrentLevel(j);
        filter->Update();
      }
    }
  }
} // end UpdateFilterNTimes

int
main(int argc, char * argv[])
{
  // Check arguments for help
  if (argc < 4)
  {
    std::cerr << "ERROR: insufficient command line arguments.\n"
              << "  inputFileName outputNameCPU outputNameGPU" << std::endl;
    return EXIT_FAILURE;
  }

  // Setup for debugging
  itk::SetupForDebugging();

  // Create and check OpenCL context
  if (!itk::CreateContext())
  {
    return EXIT_FAILURE;
  }
  const itk::OpenCLContextScopeGuard openCLContextScopeGuard{};

  /** Get the command line arguments. */
  const std::string  inputFileName = argv[1];
  const std::string  outputFileNameCPU = argv[2];
  const std::string  outputFileNameGPU = argv[3];
  const unsigned int numberOfLevels = 4;
  const bool         useMultiResolutionRescaleSchedule = true;
  const bool         useMultiResolutionSmoothingSchedule = true;
  const bool         useShrinkImageFilter = false;
  const bool         computeOnlyForCurrentLevel = true;
  const double       epsilon = 1e-3;
  const unsigned int runTimes = 5;

  std::cout << std::showpoint << std::setprecision(4);

  // Typedefs.
  const unsigned int Dimension = 3;
  using InputPixelType = float;
  using OutputPixelType = float;
  using InputImageType = itk::Image<InputPixelType, Dimension>;
  using OutputImageType = itk::Image<OutputPixelType, Dimension>;
  using GPUInputImageType = itk::GPUImage<InputPixelType, Dimension>;
  using GPUOutputImageType = itk::GPUImage<OutputPixelType, Dimension>;

  // CPU Typedefs
  using PrecisionType = float;
  using FilterType = itk::GenericMultiResolutionPyramidImageFilter<InputImageType, OutputImageType, PrecisionType>;
  using GPUFilterType =
    itk::GenericMultiResolutionPyramidImageFilter<GPUInputImageType, GPUOutputImageType, PrecisionType>;

  // Read image
  InputImageType::Pointer inputImage = itk::ReadImage<InputImageType>(inputFileName);

  // Construct the filter
  auto cpuFilter = FilterType::New();

  using RescaleScheduleType = FilterType::RescaleScheduleType;
  using SmoothingScheduleType = FilterType::SmoothingScheduleType;

  using RandomNumberGeneratorType = itk::Statistics::MersenneTwisterRandomVariateGenerator;
  RandomNumberGeneratorType::Pointer randomNum = RandomNumberGeneratorType::GetInstance();
  randomNum->SetSeed(std::mt19937::default_seed);

  RescaleScheduleType   rescaleSchedule(numberOfLevels, Dimension);
  SmoothingScheduleType smoothingSchedule(numberOfLevels, Dimension);
  double                tmp = 0.0;
  for (unsigned int i = 0; i < numberOfLevels; ++i)
  {
    for (unsigned int j = 0; j < Dimension; ++j)
    {
      tmp = randomNum->GetUniformVariate(0, 8);
      rescaleSchedule[i][j] = static_cast<unsigned int>(tmp);

      tmp = randomNum->GetUniformVariate(0, 4);
      smoothingSchedule[i][j] = tmp;
    }
  }

  cpuFilter->SetNumberOfLevels(numberOfLevels);
  cpuFilter->SetRescaleSchedule(rescaleSchedule);
  cpuFilter->SetSmoothingSchedule(smoothingSchedule);
  if (!useMultiResolutionSmoothingSchedule)
  {
    cpuFilter->SetRescaleScheduleToUnity();
  }
  if (!useMultiResolutionRescaleSchedule)
  {
    cpuFilter->SetSmoothingScheduleToZero();
  }
  cpuFilter->SetUseShrinkImageFilter(useShrinkImageFilter);
  cpuFilter->SetComputeOnlyForCurrentLevel(computeOnlyForCurrentLevel);

  std::cout << "RescaleSchedule:\n" << rescaleSchedule << "\n";
  std::cout << "SmoothingSchedule:\n" << smoothingSchedule << "\n";
  std::cout << "Testing the GenericMultiResolutionPyramidImageFilter, CPU vs GPU:\n";
  std::cout << "CPU/GPU #threads time speedup RMSE\n";
  // std::cout << "CPU/GPU factors sigmas #threads time speedup RMSE\n";

  // Time the filter, run on the CPU
  itk::TimeProbe cputimer;
  cputimer.Start();
  cpuFilter->SetInput(inputImage);
  try
  {
    UpdateFilterNTimes<FilterType>(cpuFilter, runTimes, computeOnlyForCurrentLevel);
  }
  catch (itk::ExceptionObject & e)
  {
    std::cerr << "ERROR: " << e << std::endl;
    return EXIT_FAILURE;
  }
  cputimer.Stop();

  std::cout << "CPU " << cpuFilter->GetNumberOfWorkUnits() << " " << cputimer.GetMean() / runTimes << std::endl;

  /** Write the CPU result. */
  try
  {
    itk::WriteImage(cpuFilter->GetOutput(numberOfLevels - 1), outputFileNameCPU);
  }
  catch (itk::ExceptionObject & e)
  {
    std::cerr << "ERROR: " << e << std::endl;
    return EXIT_FAILURE;
  }

  // Register object factory for GPU image and filter
  // All these filters that are constructed after this point are
  // turned into a GPU filter.
  // Note that we are not registering a
  // GPUGenericMultiResolutionPyramidImageFilter,
  // but the recursive one. We are simply using the original ITK implementation,
  // that internally uses the recursive filter. By registering the recursive
  // filter, we now automatically use it, even if it's usage is hidden by a
  // wrapper.
  using OCLImageTypes = typelist::MakeTypeList<float>::Type;
  itk::GPUImageFactory2<OCLImageTypes, OCLImageDims>::RegisterOneFactory();
  itk::GPURecursiveGaussianImageFilterFactory2<OCLImageTypes, OCLImageTypes, OCLImageDims>::RegisterOneFactory();
  itk::GPUCastImageFilterFactory2<OCLImageTypes, OCLImageTypes, OCLImageDims>::RegisterOneFactory();
  itk::GPUShrinkImageFilterFactory2<OCLImageTypes, OCLImageTypes, OCLImageDims>::RegisterOneFactory();
  itk::GPUResampleImageFilterFactory2<OCLImageTypes, OCLImageTypes, OCLImageDims>::RegisterOneFactory();
  itk::GPUIdentityTransformFactory2<OCLImageDims>::RegisterOneFactory();
  itk::GPULinearInterpolateImageFunctionFactory2<OCLImageTypes, OCLImageDims>::RegisterOneFactory();

  // Construct the filter
  // Use a try/catch, because construction of this filter will trigger
  // OpenCL compilation, which may fail.
  GPUFilterType::Pointer gpuFilter;
  try
  {
    gpuFilter = GPUFilterType::New();
    itk::ITKObjectEnableWarnings(gpuFilter.GetPointer());
  }
  catch (itk::ExceptionObject & e)
  {
    std::cerr << "ERROR: " << e << std::endl;
    return EXIT_FAILURE;
  }
  gpuFilter->SetNumberOfLevels(numberOfLevels);
  gpuFilter->SetRescaleSchedule(rescaleSchedule);
  gpuFilter->SetSmoothingSchedule(smoothingSchedule);
  if (!useMultiResolutionSmoothingSchedule)
  {
    gpuFilter->SetRescaleScheduleToUnity();
  }
  if (!useMultiResolutionRescaleSchedule)
  {
    gpuFilter->SetSmoothingScheduleToZero();
  }
  gpuFilter->SetUseShrinkImageFilter(useShrinkImageFilter);
  gpuFilter->SetComputeOnlyForCurrentLevel(computeOnlyForCurrentLevel);

  // GPU input image
  GPUInputImageType::Pointer gpuInputImage = GPUInputImageType::New();
  gpuInputImage->GraftITKImage(inputImage);
  gpuInputImage->AllocateGPU();
  gpuInputImage->GetGPUDataManager()->SetCPUBufferLock(true);
  gpuInputImage->GetGPUDataManager()->SetGPUDirtyFlag(true);
  gpuInputImage->GetGPUDataManager()->UpdateGPUBuffer();

  // Time the filter, run on the GPU
  itk::TimeProbe gputimer;
  gputimer.Start();
  gpuFilter->SetInput(gpuInputImage);
  try
  {
    UpdateFilterNTimes<GPUFilterType>(gpuFilter, runTimes, computeOnlyForCurrentLevel);
  }
  catch (itk::ExceptionObject & e)
  {
    std::cerr << "ERROR: " << e << std::endl;
    return EXIT_FAILURE;
  }
  gputimer.Stop();

  std::cout << "GPU x " << gputimer.GetMean() / runTimes << " " << cputimer.GetMean() / gputimer.GetMean();

  /** Write the GPU result. */
  try
  {
    itk::WriteImage(gpuFilter->GetOutput(numberOfLevels - 1), outputFileNameGPU);
  }
  catch (itk::ExceptionObject & e)
  {
    std::cerr << "ERROR: " << e << std::endl;
    return EXIT_FAILURE;
  }

  // Compute RMSE
  double       RMSrelative = 0.0;
  const double RMSerror = itk::ComputeRMSE<double, OutputImageType, OutputImageType>(
    cpuFilter->GetOutput(numberOfLevels - 1), gpuFilter->GetOutput(numberOfLevels - 1), RMSrelative);
  std::cout << " " << RMSerror << std::endl;

  // Check
  if (RMSerror > epsilon)
  {
    std::cerr << "ERROR: RMSE between CPU and GPU result larger than expected" << std::endl;
    return EXIT_FAILURE;
  }

  // End program.
  return EXIT_SUCCESS;
} // end main()
