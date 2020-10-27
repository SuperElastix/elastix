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

// ITK include files
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionConstIterator.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"
#include "itkTimeProbe.h"

#include <iomanip> // setprecision, etc.

//------------------------------------------------------------------------------
// This test compares the CPU with the GPU version of the
// GenericMultiResolutionPyramidImageFilter.
// The filter takes an input image and produces an output image.
// We compare the CPU and GPU output image write RMSE and speed.

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
  const unsigned int                             Dimension = 3;
  typedef float                                  InputPixelType;
  typedef float                                  OutputPixelType;
  typedef itk::Image<InputPixelType, Dimension>  InputImageType;
  typedef itk::Image<OutputPixelType, Dimension> OutputImageType;

  // CPU Typedefs
  typedef itk::GenericMultiResolutionPyramidImageFilter<InputImageType, OutputImageType> FilterType;
  typedef itk::ImageFileReader<InputImageType>                                           ReaderType;
  typedef itk::ImageFileWriter<OutputImageType>                                          WriterType;

  // Reader
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(inputFileName);
  reader->Update();

  // Construct the filter
  FilterType::Pointer cpuFilter = FilterType::New();

  typedef FilterType::RescaleScheduleType   RescaleScheduleType;
  typedef FilterType::SmoothingScheduleType SmoothingScheduleType;

  typedef itk::Statistics::MersenneTwisterRandomVariateGenerator RandomNumberGeneratorType;
  RandomNumberGeneratorType::Pointer                             randomNum = RandomNumberGeneratorType::GetInstance();

  RescaleScheduleType   rescaleSchedule(numberOfLevels, Dimension);
  SmoothingScheduleType smoothingSchedule(numberOfLevels, Dimension);
  double                tmp = 0.0;
  for (unsigned int i = 0; i < numberOfLevels; i++)
  {
    for (unsigned int j = 0; j < Dimension; j++)
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
  for (unsigned int i = 0; i < runTimes; i++)
  {
    cpuFilter->SetInput(reader->GetOutput());

    if (!computeOnlyForCurrentLevel)
    {
      try
      {
        cpuFilter->Update();
      }
      catch (itk::ExceptionObject & e)
      {
        std::cerr << "ERROR: " << e << std::endl;
        itk::ReleaseContext();
        return EXIT_FAILURE;
      }
      cpuFilter->Modified();
    }
    else
    {
      for (unsigned int j = 0; j < cpuFilter->GetNumberOfLevels(); j++)
      {
        cpuFilter->SetCurrentLevel(j);
        try
        {
          cpuFilter->Update();
        }
        catch (itk::ExceptionObject & e)
        {
          std::cerr << "ERROR: " << e << std::endl;
          itk::ReleaseContext();
          return EXIT_FAILURE;
        }
        // Modify the filter, only not the last iteration
        if (i != runTimes - 1)
        {
          cpuFilter->Modified();
        }
      }
    }
  }
  cputimer.Stop();

  std::cout << "CPU " << cpuFilter->GetNumberOfWorkUnits() << " " << cputimer.GetMean() / runTimes << std::endl;

  /** Write the CPU result. */
  WriterType::Pointer writer = WriterType::New();
  writer->SetInput(cpuFilter->GetOutput(numberOfLevels - 1));
  writer->SetFileName(outputFileNameCPU.c_str());
  try
  {
    writer->Update();
  }
  catch (itk::ExceptionObject & e)
  {
    std::cerr << "ERROR: " << e << std::endl;
    itk::ReleaseContext();
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
  typedef typelist::MakeTypeList<float>::Type OCLImageTypes;
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
  FilterType::Pointer gpuFilter;
  try
  {
    gpuFilter = FilterType::New();
    itk::ITKObjectEnableWarnings(gpuFilter.GetPointer());
  }
  catch (itk::ExceptionObject & e)
  {
    std::cerr << "ERROR: " << e << std::endl;
    itk::ReleaseContext();
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

  // Also need to re-construct the image reader, so that it now
  // reads a GPUImage instead of a normal image.
  // Otherwise, you will get an exception when running the GPU filter:
  // "ERROR: The GPU InputImage is NULL. Filter unable to perform."
  ReaderType::Pointer gpuReader = ReaderType::New();
  gpuReader->SetFileName(inputFileName);

  // \todo: If the following line is uncommented something goes wrong with
  // the ITK pipeline synchronization.
  // Something is still read in, but I guess it is not properly copied to
  // the GPU. The output of the shrink filter is then bogus.
  // The following line is however not needed in a pure CPU implementation.
  gpuReader->Update();

  // Time the filter, run on the GPU
  itk::TimeProbe gputimer;
  gputimer.Start();
  for (unsigned int i = 0; i < runTimes; i++)
  {
    gpuFilter->SetInput(gpuReader->GetOutput());

    if (!computeOnlyForCurrentLevel)
    {
      try
      {
        gpuFilter->Update();
      }
      catch (itk::ExceptionObject & e)
      {
        std::cerr << "ERROR: " << e << std::endl;
        itk::ReleaseContext();
        return EXIT_FAILURE;
      }
      // Modify the filter, only not the last iteration
      if (i != runTimes - 1)
      {
        gpuFilter->Modified();
      }
    }
    else
    {
      for (unsigned int j = 0; j < gpuFilter->GetNumberOfLevels(); j++)
      {
        gpuFilter->SetCurrentLevel(j);
        try
        {
          gpuFilter->Update();
        }
        catch (itk::ExceptionObject & e)
        {
          std::cerr << "ERROR: " << e << std::endl;
          itk::ReleaseContext();
          return EXIT_FAILURE;
        }
        // Modify the filter, only not the last iteration
        if (i != runTimes - 1)
        {
          gpuFilter->Modified();
        }
      }
    }
  }
  gputimer.Stop();

  std::cout << "GPU x " << gputimer.GetMean() / runTimes << " " << cputimer.GetMean() / gputimer.GetMean();

  /** Write the GPU result. */
  WriterType::Pointer gpuWriter = WriterType::New();
  gpuWriter->SetInput(gpuFilter->GetOutput(numberOfLevels - 1));
  gpuWriter->SetFileName(outputFileNameGPU.c_str());
  try
  {
    gpuWriter->Update();
  }
  catch (itk::ExceptionObject & e)
  {
    std::cerr << "ERROR: " << e << std::endl;
    itk::ReleaseContext();
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
    itk::ReleaseContext();
    return EXIT_FAILURE;
  }

  // End program.
  itk::ReleaseContext();
  return EXIT_SUCCESS;
} // end main()
