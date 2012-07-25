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
#include "itkGPUBSplineDecompositionImageFilter.h"
#include "itkGPUExplicitSynchronization.h"

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "itkTimeProbe.h"
#include "itkOpenCLUtil.h" // IsGPUAvailable()


//------------------------------------------------------------------------------

int main( int argc, char * argv[] )
{
  // Check arguments for help
  if( argc < 2 )
  {
    std::cerr << "ERROR: insufficient command line arguments.\n"
      << "  inputFileName" << std::endl;
    return EXIT_FAILURE;
  }

  // Check for GPU
  if( !itk::IsGPUAvailable() )
  {
    std::cerr << "ERROR: OpenCL-enabled GPU is not present." << std::endl;
    return EXIT_FAILURE;
  }

  /** Get the command line arguments. */
  std::string inputFileName = argv[1];
  std::string outputDirectory = argv[2];
  std::string baseName = inputFileName.substr( 0, inputFileName.rfind( "." ) );
  std::string outputFileNameCPU = outputDirectory + "/" + baseName + "-out-cpu.mha";
  std::string outputFileNameGPU = outputDirectory + "/" + baseName + "-out-gpu.mha";
  const unsigned int splineOrder = 3;
  const double eps = 1e-3;
  const unsigned int runTimes = 5;

  // Typedefs.
  const unsigned int  Dimension = 3;
  typedef float       PixelType;
  typedef itk::Image<PixelType, Dimension>  ImageType;

  // CPU Typedefs
  typedef itk::BSplineDecompositionImageFilter<ImageType, ImageType> FilterType;
  typedef itk::ImageFileReader<ImageType> ReaderType;
  typedef itk::ImageFileWriter<ImageType> WriterType;

  // Reader
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName( inputFileName );
  reader->Update();

  // Construct the filter
  FilterType::Pointer filter = FilterType::New();
  filter->SetSplineOrder( splineOrder );

  // Time the filter, run on the CPU
  itk::TimeProbe cputimer;
  cputimer.Start();
  for( unsigned int i = 0; i < runTimes; i++ )
  {
    filter->SetInput( reader->GetOutput() );
    try{ filter->Update(); }
    catch( itk::ExceptionObject & e )
    {
      std::cerr << "ERROR: " << e << std::endl;
      return EXIT_FAILURE;
    }
    filter->Modified();
  }
  cputimer.Stop();

  std::cout << "CPU " << filter->GetNameOfClass()
    << " took " << cputimer.GetMean() / runTimes << " seconds with "
    << filter->GetNumberOfThreads() << " threads." << std::endl;

  // Copy the result

  // Register object factory for GPU image and filter
  // All these filters that are constructed after this point are
  // turned into a GPU filter.
  itk::ObjectFactoryBase::RegisterFactory( itk::GPUImageFactory::New() );
  itk::ObjectFactoryBase::RegisterFactory( itk::GPUBSplineDecompositionImageFilterFactory::New() );
  
  // Construct the filter
  // Use a try/catch, because construction of this filter will trigger
  // OpenCL compilation, which may fail.
  FilterType::Pointer gpuFilter;
  try{ gpuFilter= FilterType::New(); }
  catch( itk::ExceptionObject &e )
  {
    std::cerr << "ERROR: " << e << std::endl;
    return EXIT_FAILURE;
  }
  gpuFilter->SetSplineOrder( splineOrder );

  // Also need to re-construct the image reader, so that it now
  // reads a GPUImage instead of a normal image.
  // Otherwise, you will get an exception when running the GPU filter:
  // "ERROR: The GPU InputImage is NULL. Filter unable to perform."
  ReaderType::Pointer gpuReader = ReaderType::New();
  gpuReader->SetFileName( inputFileName );

  // Time the filter, run on the GPU
  itk::TimeProbe gputimer;
  gputimer.Start();
  for( unsigned int i = 0; i < runTimes; i++ )
  {
    std::cerr << i << std::endl;
    gpuFilter->SetInput( gpuReader->GetOutput() );
    // I get an OpenCL Error: CL_INVALID_WORK_GROUP_SIZE on my NVidia FX1700 with OPenCL 1.0
    try{ gpuFilter->Update(); }
    catch( itk::ExceptionObject &e )
    {
      std::cerr << "ERROR: " << e << std::endl;
      return EXIT_FAILURE;
    }
    // Due to some bug in the ITK synchronisation we now manually
    // copy the result from GPU to CPU, without calling Update() again,
    // and not clearing GPU memory afterwards.
    itk::GPUExplicitSync<FilterType, ImageType>( gpuFilter, false, false );
    //itk::GPUExplicitSync<FilterType, ImageType>( gpuFilter, false, true ); // crashes!
    gpuFilter->Modified();
  }
  // GPU buffer has not been copied yet, so we have to make manual update
  //itk::GPUExplicitSync<GPUFilterType, OutputImageType>( GPUFilter, false );
  gputimer.Stop();

  std::cout << "GPU " << gpuFilter->GetNameOfClass()
    << " took " << gputimer.GetMean() / runTimes
    << " seconds" << std::endl;

  //// RMS Error check
  //const double epsilon = 0.01;
  //float diff = 0.0;
  //unsigned int nPix = 0;

  //  itk::ImageRegionIterator<OutputImageType> cit( CPUFilter->GetOutput(),
  //    CPUFilter->GetOutput()->GetLargestPossibleRegion() );
  //  itk::ImageRegionIterator<OutputImageType> git( GPUFilter->GetOutput(),
  //    GPUFilter->GetOutput()->GetLargestPossibleRegion() );
  //  for( cit.GoToBegin(), git.GoToBegin(); !cit.IsAtEnd(); ++cit, ++git )
  //  {
  //    float c = (float)(cit.Get());
  //    float g = (float)(git.Get());
  //    float err = vnl_math_abs( c - g );
  //    //if(err > epsilon)
  //    //  std::cout << "CPU : " << (double)(cit.Get()) << ", GPU : " << (double)(git.Get()) << std::endl;
  //    diff += err*err;
  //    nPix++;
  //  }
  //}

  //float RMSError = 0.0;
  //if( !_parameters.skipCPU && !_parameters.skipGPU && !updateException )
  //{
  //  RMSError = vcl_sqrt( diff / (float)nPix );
  //  std::cout << "RMS Error: " << std::fixed << std::setprecision(8) << RMSError << std::endl;
  //}
  //bool testPassed = false;
  //if( !updateException )
  //{
  //  testPassed = ( RMSError <= _parameters.RMSError );
  //}

  //// Write output
  //if( _parameters.outputWrite )
  //{
  //  if( !_parameters.skipCPU )
  //  {
  //    // Write output CPU image
  //    typename CPUWriterType::Pointer writerCPU = CPUWriterType::New();
  //    writerCPU->SetInput( CPUFilter->GetOutput() );
  //    writerCPU->SetFileName( _parameters.outputFileNames[0] );
  //    writerCPU->Update();
  //  }

  //  if( !_parameters.skipGPU && !updateException )
  //  {
  //    // Write output GPU image
  //    typename GPUWriterType::Pointer writerGPU = GPUWriterType::New();
  //    writerGPU->SetInput( GPUFilter->GetOutput() );
  //    writerGPU->SetFileName( _parameters.outputFileNames[1] );
  //    writerGPU->Update();
  //  }
  //}




  // End program.
  return EXIT_SUCCESS;

} // end main()
