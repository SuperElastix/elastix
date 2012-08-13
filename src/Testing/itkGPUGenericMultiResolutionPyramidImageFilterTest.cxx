/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

// GPU include files
#include "itkGenericMultiResolutionPyramidImageFilter.h"
#include "itkGPURecursiveGaussianImageFilter.h"
#include "itkGPUCastImageFilter.h"
#include "itkGPUResampleImageFilter.h"
#include "itkGPUIdentityTransform.h"
#include "itkGPULinearInterpolateImageFunction.h"
#include "itkGPUShrinkImageFilter.h"
#include "itkGPUExplicitSynchronization.h"

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionConstIterator.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"

#include "itkTimeProbe.h"
#include "itkOpenCLUtil.h" // IsGPUAvailable()
#include <iomanip> // setprecision, etc.

// Helper function
template<class ImageType>
double ComputeRMSE( ImageType * cpuImage, ImageType * gpuImage )
{
  itk::ImageRegionConstIterator<ImageType> cit(
    cpuImage, cpuImage->GetLargestPossibleRegion() );
  itk::ImageRegionConstIterator<ImageType> git(
    gpuImage, gpuImage->GetLargestPossibleRegion() );

  double rmse = 0.0;
  for( cit.GoToBegin(), git.GoToBegin(); !cit.IsAtEnd(); ++cit, ++git )
  {
    double err = static_cast<double>( cit.Get() ) - static_cast<double>( git.Get() );
    rmse += err * err;
  }
  rmse = vcl_sqrt( rmse / cpuImage->GetLargestPossibleRegion().GetNumberOfPixels() );
  return rmse;
} // end ComputeRMSE()

//------------------------------------------------------------------------------
// This test compares the CPU with the GPU version of the GenericMultiResolutionPyramidImageFilter.
// The filter takes an input image and produces an output image.
// We compare the CPU and GPU output image write RMSE and speed.

int main( int argc, char * argv[] )
{
  // Check arguments for help
  if( argc < 4 )
  {
    std::cerr << "ERROR: insufficient command line arguments.\n"
      << "  inputFileName outputNameCPU outputNameGPU" << std::endl;
    return EXIT_FAILURE;
  }

  // Check for GPU
  if( !itk::IsGPUAvailable() )
  {
    std::cerr << "ERROR: OpenCL-enabled GPU is not present." << std::endl;
    return EXIT_FAILURE;
  }

  /** Get the command line arguments. */
  const std::string inputFileName = argv[1];
  const std::string outputFileNameCPU = argv[2];
  const std::string outputFileNameGPU = argv[3];
  const unsigned int numberOfLevels = 3;
  const bool useMultiResolutionRescaleSchedule = true;
  const bool useMultiResolutionSmoothingSchedule = true;
  const bool useShrinkImageFilter = true;
  const bool computeOnlyForCurrentLevel = true;
  const double epsilon = 1e-3;
  const unsigned int runTimes = 5;

  std::cout << std::showpoint << std::setprecision( 4 );

  // Typedefs.
  const unsigned int  Dimension = 3;
  typedef short       InputPixelType;
  typedef float       OutputPixelType;
  typedef itk::Image<InputPixelType, Dimension>   InputImageType;
  typedef itk::Image<OutputPixelType, Dimension>  OutputImageType;

  // CPU Typedefs
  typedef itk::GenericMultiResolutionPyramidImageFilter<
    InputImageType, OutputImageType >             FilterType;
  typedef itk::ImageFileReader<InputImageType>    ReaderType;
  typedef itk::ImageFileWriter<OutputImageType>   WriterType;

  // Reader
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName( inputFileName );
  reader->Update();

  // Construct the filter
  FilterType::Pointer filter = FilterType::New();

  typedef FilterType::RescaleScheduleType   RescaleScheduleType;
  typedef FilterType::SmoothingScheduleType SmoothingScheduleType;

  typedef itk::Statistics::MersenneTwisterRandomVariateGenerator RandomNumberGeneratorType;
  RandomNumberGeneratorType::Pointer randomNum = RandomNumberGeneratorType::GetInstance();

  RescaleScheduleType   rescaleSchedule(   numberOfLevels, Dimension );
  SmoothingScheduleType smoothingSchedule( numberOfLevels, Dimension );
  double tmp = 0.0;
  for( unsigned int i = 0; i < numberOfLevels; i++ )
  {
    for( unsigned int j = 0; j < Dimension; j++ )
    {
      tmp = randomNum->GetUniformVariate( 0, 8 );
      rescaleSchedule[ i ][ j ] = static_cast<unsigned int>( tmp );

      tmp = randomNum->GetUniformVariate( 0, 4 );
      smoothingSchedule[ i ][ j ] = tmp;
    }
  }

  filter->SetNumberOfLevels( numberOfLevels );
  filter->SetRescaleSchedule( rescaleSchedule );
  filter->SetSmoothingSchedule( smoothingSchedule );
  filter->SetUseMultiResolutionRescaleSchedule( useMultiResolutionRescaleSchedule );
  filter->SetUseMultiResolutionSmoothingSchedule( useMultiResolutionSmoothingSchedule );
  filter->SetUseShrinkImageFilter( useShrinkImageFilter );
  filter->SetComputeOnlyForCurrentLevel( computeOnlyForCurrentLevel );

  std::cout << "RescaleSchedule:\n" << rescaleSchedule << "\n";
  std::cout << "SmoothingSchedule:\n" << smoothingSchedule << "\n";
  std::cout << "Testing the GenericMultiResolutionPyramidImageFilter, CPU vs GPU:\n";
  std::cout << "CPU/GPU #threads time speedup RMSE\n";
  //std::cout << "CPU/GPU factors sigmas #threads time speedup RMSE\n";

  // Time the filter, run on the CPU
  itk::TimeProbe cputimer;
  cputimer.Start();
  for( unsigned int i = 0; i < runTimes; i++ )
  {
    filter->SetInput( reader->GetOutput() );

    if( !computeOnlyForCurrentLevel )
    {
      try{ filter->Update(); }
      catch( itk::ExceptionObject & e )
      {
        std::cerr << "ERROR: " << e << std::endl;
        return EXIT_FAILURE;
      }
      filter->Modified();
    }
    else
    {
      for( unsigned int j = 0; j < filter->GetNumberOfLevels(); j++ )
      {
        filter->SetCurrentLevel( j );
        try{ filter->Update(); }
        catch( itk::ExceptionObject & e )
        {
          std::cerr << "ERROR: " << e << std::endl;
          return EXIT_FAILURE;
        }
        // Modify the filter, only not the last iteration
        if( i != runTimes - 1 )
        {
          filter->Modified();
        }
      }
    }
  }
  cputimer.Stop();

  std::cout << "CPU "
    << filter->GetNumberOfThreads()
    << " " << cputimer.GetMean() / runTimes << std::endl;

  /** Write the CPU result. */
  WriterType::Pointer writer = WriterType::New();
  writer->SetInput( filter->GetOutput( numberOfLevels - 1 ) );
  writer->SetFileName( outputFileNameCPU.c_str() );
  try{ writer->Update(); }
  catch( itk::ExceptionObject & e )
  {
    std::cerr << "ERROR: " << e << std::endl;
    return EXIT_FAILURE;
  }

  // Register object factory for GPU image and filter
  // All these filters that are constructed after this point are
  // turned into a GPU filter.
  // Note that we are not registering a GPUGenericMultiResolutionPyramidImageFilter,
  // but the recursive one. We are simply using the original ITK implementation,
  // that internally uses the recursive filter. By registering the recursive
  // filter, we now automatically use it, even if it's usage is hidden by a wrapper.
  itk::ObjectFactoryBase::RegisterFactory( itk::GPUImageFactory::New() );
  itk::ObjectFactoryBase::RegisterFactory( itk::GPURecursiveGaussianImageFilterFactory::New() );
  itk::ObjectFactoryBase::RegisterFactory( itk::GPUCastImageFilterFactory::New() );
  itk::ObjectFactoryBase::RegisterFactory( itk::GPUShrinkImageFilterFactory::New() );
  itk::ObjectFactoryBase::RegisterFactory( itk::GPUResampleImageFilterFactory::New() );
  itk::ObjectFactoryBase::RegisterFactory( itk::GPUIdentityTransformFactory::New() );
  itk::ObjectFactoryBase::RegisterFactory( itk::GPULinearInterpolateImageFunctionFactory::New() );

  // Construct the filter
  // Use a try/catch, because construction of this filter will trigger
  // OpenCL compilation, which may fail.
  FilterType::Pointer gpuFilter;
  try{ gpuFilter = FilterType::New(); }
  catch( itk::ExceptionObject & e )
  {
    std::cerr << "ERROR: " << e << std::endl;
    return EXIT_FAILURE;
  }
  gpuFilter->SetNumberOfLevels( numberOfLevels );
  gpuFilter->SetRescaleSchedule( rescaleSchedule );
  gpuFilter->SetSmoothingSchedule( smoothingSchedule );
  gpuFilter->SetUseMultiResolutionRescaleSchedule( useMultiResolutionRescaleSchedule );
  gpuFilter->SetUseMultiResolutionSmoothingSchedule( useMultiResolutionSmoothingSchedule );
  gpuFilter->SetUseShrinkImageFilter( useShrinkImageFilter );
  gpuFilter->SetComputeOnlyForCurrentLevel( computeOnlyForCurrentLevel );

  // Also need to re-construct the image reader, so that it now
  // reads a GPUImage instead of a normal image.
  // Otherwise, you will get an exception when running the GPU filter:
  // "ERROR: The GPU InputImage is NULL. Filter unable to perform."
  ReaderType::Pointer gpuReader = ReaderType::New();
  gpuReader->SetFileName( inputFileName );

  // \todo: If the following line is uncommented something goes wrong with
  // the ITK pipeline synchronisation.
  // Something is still read in, but I guess it is not properly copied to
  // the GPU. The output of the shrink filter is then bogus.
  // The following line is however not needed in a pure CPU implementation.
  gpuReader->Update();

  // Time the filter, run on the GPU
  itk::TimeProbe gputimer;
  gputimer.Start();
  for( unsigned int i = 0; i < runTimes; i++ )
  {
    gpuFilter->SetInput( gpuReader->GetOutput() );

    if( !computeOnlyForCurrentLevel )
    {
      try{ gpuFilter->Update(); }
      catch( itk::ExceptionObject & e )
      {
        std::cerr << "ERROR: " << e << std::endl;
        return EXIT_FAILURE;
      }
      itk::GPUExplicitSync<FilterType, OutputImageType>( gpuFilter, false, false );
      // Modify the filter, only not the last iteration
      if( i != runTimes - 1 )
      {
        gpuFilter->Modified();
      }
    }
    else
    {
      for( unsigned int j = 0; j < gpuFilter->GetNumberOfLevels(); j++ )
      {
        gpuFilter->SetCurrentLevel( j );
        try{ gpuFilter->Update(); }
        catch( itk::ExceptionObject & e )
        {
          std::cerr << "ERROR: " << e << std::endl;
          return EXIT_FAILURE;
        }
        itk::GPUExplicitSync<FilterType, OutputImageType>( gpuFilter, false, false );
        // Modify the filter, only not the last iteration
        if( i != runTimes - 1 )
        {
          gpuFilter->Modified();
        }
      }
    }
  }
  gputimer.Stop();

  std::cout << "GPU x "
    << gputimer.GetMean() / runTimes
    << " " << cputimer.GetMean() / gputimer.GetMean();

  /** Write the GPU result. */
  WriterType::Pointer gpuWriter = WriterType::New();
  gpuWriter->SetInput( gpuFilter->GetOutput( numberOfLevels - 1 ) );
  gpuWriter->SetFileName( outputFileNameGPU.c_str() );
  try{ gpuWriter->Update(); }
  catch( itk::ExceptionObject & e )
  {
    std::cerr << "ERROR: " << e << std::endl;
    return EXIT_FAILURE;
  }

  // Compute RMSE
  const double rmse = ComputeRMSE<OutputImageType>(
    filter->GetOutput( numberOfLevels - 1 ), gpuFilter->GetOutput( numberOfLevels - 1 ) );
  std::cout << " " << rmse << std::endl;

  // Check
  if( rmse > epsilon )
  {
    std::cerr << "ERROR: RMSE between CPU and GPU result larger than expected" << std::endl;
    return EXIT_FAILURE;
  }

  // End program.
  return EXIT_SUCCESS;

} // end main()
