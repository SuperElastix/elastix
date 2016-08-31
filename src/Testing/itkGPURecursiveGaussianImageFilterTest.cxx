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

// GPU include files
#include "itkGPURecursiveGaussianImageFilter.h"

// ITK include files
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionConstIterator.h"
#include "itkTimeProbe.h"

#include <iomanip> // setprecision, etc.

//------------------------------------------------------------------------------
// This test compares the CPU with the GPU version of the
// RecursiveGaussianImageFilter.
// The filter takes an input image and produces an output image.
// We compare the CPU and GPU output image write RMSE and speed.

int
main( int argc, char * argv[] )
{
  const unsigned int ImageDimension = 3; // 2

  typedef float                                            InputPixelType;
  typedef float                                            OutputPixelType;
  typedef itk::GPUImage< InputPixelType,  ImageDimension > InputImageType;
  typedef itk::GPUImage< OutputPixelType, ImageDimension > OutputImageType;

  typedef itk::RecursiveGaussianImageFilter
    < InputImageType, OutputImageType > CPUFilterType;
  typedef itk::GPURecursiveGaussianImageFilter
    < InputImageType, OutputImageType > GPUFilterType;

  typedef itk::ImageFileReader< InputImageType >  ReaderType;
  typedef itk::ImageFileWriter< OutputImageType > WriterType;

  if( argc <  3 )
  {
    std::cerr << "ERROR: missing arguments" << std::endl;
    std::cerr << "  inputfile outputfile " << std::endl;
    return EXIT_FAILURE;
  }

  // Setup for debugging
  itk::SetupForDebugging();

  // Create and check OpenCL context
  if( !itk::CreateContext() )
  {
    return EXIT_FAILURE;
  }

  // Some hard-coded testing options
  const std::string  inputFileName     = argv[ 1 ];
  const std::string  outputFileNameCPU = argv[ 2 ];
  const std::string  outputFileNameGPU = argv[ 3 ];
  const double       sigma             = 3.0;
  unsigned int       direction         = 0;
  const double       epsilon           = 0.01;
  const unsigned int maximumNumberOfThreads
    = itk::MultiThreader::GetGlobalDefaultNumberOfThreads();

  std::cout << std::showpoint << std::setprecision( 4 );

  // Reader
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName( inputFileName );
  try
  {
    reader->Update();
  }
  catch( itk::ExceptionObject & excp )
  {
    std::cerr << "ERROR: " << excp << std::endl;
    itk::ReleaseContext();
    return EXIT_FAILURE;
  }

  itk::TimeProbe cputimer;
  itk::TimeProbe gputimer;

  CPUFilterType::Pointer cpuFilter = CPUFilterType::New();

  // Test 1~n threads for CPU
  // Speed CPU vs GPU
  std::cout << "Testing the Recursive Gaussian filter, CPU vs GPU:\n";
  std::cout << "CPU/GPU sigma direction #threads time speedup RMSE\n";

  for( unsigned int nThreads = 1; nThreads <= maximumNumberOfThreads; nThreads++ )
  {
    // Test CPU
    cputimer.Start();
    cpuFilter->SetNumberOfThreads( nThreads );
    cpuFilter->SetInput( reader->GetOutput() );
    cpuFilter->SetSigma( sigma );
    cpuFilter->SetDirection( direction );
    cpuFilter->Update();
    cputimer.Stop();

    std::cout << "CPU " << sigma << " " << direction << " " << nThreads
              << " " << cputimer.GetMean() << std::endl;
  }

  /** Write the CPU result. */
  WriterType::Pointer writer = WriterType::New();
  writer->SetInput( cpuFilter->GetOutput() );
  writer->SetFileName( outputFileNameCPU.c_str() );
  try
  {
    writer->Update();
  }
  catch( itk::ExceptionObject & e )
  {
    std::cerr << "ERROR: " << e << std::endl;
    itk::ReleaseContext();
    return EXIT_FAILURE;
  }

  // Construct the filter
  // Use a try/catch, because construction of this filter will trigger
  // OpenCL compilation, which may fail.
  GPUFilterType::Pointer gpuFilter;
  try
  {
    gpuFilter = GPUFilterType::New();
    itk::ITKObjectEnableWarnings( gpuFilter.GetPointer() );
  }
  catch( itk::ExceptionObject & e )
  {
    std::cerr << "ERROR: " << e << std::endl;
    itk::ReleaseContext();
    return EXIT_FAILURE;
  }

  // Test GPU
  gputimer.Start();
  gpuFilter->SetInput( reader->GetOutput() );
  gpuFilter->SetSigma( sigma );
  gpuFilter->SetDirection( direction );
  gpuFilter->Update();
  gputimer.Stop();

  std::cout << "GPU " << sigma << " " << direction << " x "
            << gputimer.GetMean()
            << " " << cputimer.GetMean() / gputimer.GetMean();

  /** Write the GPU result. */
  WriterType::Pointer gpuWriter = WriterType::New();
  gpuWriter->SetInput( gpuFilter->GetOutput() );
  gpuWriter->SetFileName( outputFileNameGPU.c_str() );
  try
  {
    gpuWriter->Update();
  }
  catch( itk::ExceptionObject & e )
  {
    std::cerr << "ERROR: " << e << std::endl;
    itk::ReleaseContext();
    return EXIT_FAILURE;
  }

  // Compute RMSE
  double       RMSrelative = 0.0;
  const double RMSerror    = itk::ComputeRMSE< double, OutputImageType, OutputImageType >
      ( cpuFilter->GetOutput(), gpuFilter->GetOutput(), RMSrelative );
  std::cout << " " << RMSerror << std::endl;

  // Check
  if( RMSerror > epsilon )
  {
    std::cerr << "ERROR: RMSE between CPU and GPU result larger than expected" << std::endl;
    itk::ReleaseContext();
    return EXIT_FAILURE;
  }

  std::cout << "\n\nTesting directions switch, CPU vs GPU:\n";
  std::cout << "CPU/GPU sigma direction #threads time speedup RMSE\n";

  // Check directions
  for( direction = 0; direction < ImageDimension; direction++ )
  {
    cputimer.Start();
    cpuFilter->SetNumberOfThreads( maximumNumberOfThreads );
    cpuFilter->SetInput( reader->GetOutput() );
    cpuFilter->SetSigma( sigma );
    cpuFilter->SetDirection( direction );
    cpuFilter->Modified();
    cpuFilter->Update();
    cputimer.Stop();

    std::cout << "CPU " << sigma << " " << direction << " " << maximumNumberOfThreads
              << " " << cputimer.GetMean() << std::endl;

    // Test GPU
    gputimer.Start();
    gpuFilter->SetInput( reader->GetOutput() );
    gpuFilter->SetSigma( sigma );
    gpuFilter->SetDirection( direction );
    gpuFilter->Modified();
    gpuFilter->Update();
    gputimer.Stop();

    std::cout << "GPU " << sigma << " " << direction << " x "
              << gputimer.GetMean()
              << " " << cputimer.GetMean() / gputimer.GetMean();

    // Compute RMSE
    double       RMSrelative = 0.0;
    const double RMSerror    = itk::ComputeRMSE< double, OutputImageType, OutputImageType >
        ( cpuFilter->GetOutput(), gpuFilter->GetOutput(), RMSrelative );
    std::cout << " " << RMSerror << std::endl;

    // Check
    if( RMSerror > epsilon )
    {
      std::cerr << "ERROR: RMSE between CPU and GPU result larger than expected" << std::endl;
      itk::ReleaseContext();
      return EXIT_FAILURE;
    }
  }

  // End program.
  itk::ReleaseContext();
  return EXIT_SUCCESS;
} // end main()
