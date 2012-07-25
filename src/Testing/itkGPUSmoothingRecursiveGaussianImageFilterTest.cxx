/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#include "itkCommandLineArgumentParser.h"
#include "CommandLineArgumentHelper.h"
#include "LoggerHelper.h"

#pragma warning(push)
// warning C4996: 'std::copy': Function call with parameters that may be unsafe
// - this call relies on the caller to check that the passed values are correct.
// To disable this warning, use -D_SCL_SECURE_NO_WARNINGS. See documentation on
// how to use Visual C++ 'Checked Iterators'
#pragma warning(disable:4996)
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#pragma warning(pop)

#include "itkTimeProbe.h"
#include "itkSmoothingRecursiveGaussianImageFilter.h"

// GPU include files
#include "itkGPURecursiveGaussianImageFilter.h"
#include "itkGPUCastImageFilter.h"
#include "itkGPUExplicitSynchronization.h"

void PrintHelp();
//------------------------------------------------------------------------------
/** run: A macro to call a function. */
#define run(function,type0,type1,dim) \
  if (ComponentType == #type0 && Dimension == dim) \
{ \
  typedef itk::Image< type0, dim > InputImageType; \
  typedef itk::Image< type1, dim > OutputImageType; \
  supported = true; \
  result = function< InputImageType, OutputImageType >( parameters ); \
}

//------------------------------------------------------------------------------
namespace {
  class Parameters
  {
  public:
    // Constructor
    Parameters()
    {
      useCompression = false;
      outputWrite = true;
      outputLog = true;
      GPUEnable = 0;
      RMSError = 0.0;
      runTimes = 1;
      skipCPU = false;
      skipGPU = false;

      sigma = 3.0;

      // Files
      logFileName = "CPUGPULog.txt";
    }

    bool useCompression;
    bool outputWrite;
    bool outputLog;
    bool skipCPU;
    bool skipGPU;
    int GPUEnable;
    float RMSError;
    unsigned int runTimes;

    // Files
    std::string inputFileName;
    std::vector<std::string> outputFileNames;
    std::string logFileName;

    // Filter
    double sigma;
  };
}

//------------------------------------------------------------------------------
template<class InputImageType, class OutputImageType>
int ProcessImage(const Parameters &_parameters);

// Testing GPU Rescale Image Filter

// 2D:
// -in D:\\work\\elastix-ext\\ITK4OpenCL\\data\\image-256x256-2D.mha -out D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test3\\image-256x256-2D-out-cpu.mha D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test3\\image-256x256-2D-out-gpu.mha -rsm 0.002 -gpu
// 3D:
// -in D:\\work\\elastix-ext\\ITK4OpenCL\\data\\image-256x256x256-3D.mha -out D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test3\\image-256x256x256-3D-out-cpu.mha D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test3\\image-256x256x256-3D-out-gpu.mha -rsm 0.002 -gpu
int main(int argc, char *argv[])
{
  // Check arguments for help
  if ( argc < 5 )
  {
    PrintHelp();
    return EXIT_FAILURE;
  }

  // Check for GPU
  if(!itk::IsGPUAvailable())
  {
    std::cerr << "OpenCL-enabled GPU is not present." << std::endl;
    return EXIT_FAILURE;
  }

  // Create a command line argument parser
  itk::CommandLineArgumentParser::Pointer parser = itk::CommandLineArgumentParser::New();
  parser->SetCommandLineArguments(argc, argv);

  // Create parameters class.
  Parameters parameters;

  // Get file names arguments
  const bool retin = parser->GetCommandLineArgument("-in", parameters.inputFileName);
  parameters.outputFileNames.push_back(parameters.inputFileName.substr(0, parameters.inputFileName.rfind( "." ))+"-out-cpu.mha");
  parameters.outputFileNames.push_back(parameters.inputFileName.substr(0, parameters.inputFileName.rfind( "." ))+"-out-gpu.mha");
  parser->GetCommandLineArgument("-out", parameters.outputFileNames);
  parameters.outputWrite = !(parser->ArgumentExists( "-nooutput" ));
  parser->GetCommandLineArgument("-outlog", parameters.logFileName);
  parser->GetCommandLineArgument("-runtimes", parameters.runTimes);

  parser->GetCommandLineArgument("-rsm", parameters.RMSError);
  parameters.GPUEnable = parser->ArgumentExists("-gpu");
  parameters.skipCPU = parser->ArgumentExists("-skipcpu");
  parameters.skipGPU = parser->ArgumentExists("-skipgpu");

  // Filter parameters
  parser->GetCommandLineArgument("-sigma", parameters.sigma);

  // Threads.
  unsigned int maximumNumberOfThreads = itk::MultiThreader::GetGlobalDefaultNumberOfThreads();
  parser->GetCommandLineArgument( "-threads", maximumNumberOfThreads );
  itk::MultiThreader::SetGlobalMaximumNumberOfThreads( maximumNumberOfThreads );

  // Determine image properties.
  std::string ComponentType = "short";
  std::string PixelType; //we don't use this
  unsigned int Dimension = 2;
  unsigned int NumberOfComponents = 1;
  std::vector<unsigned int> imagesize( Dimension, 0 );
  int retgip = GetImageProperties(
    parameters.inputFileName,
    PixelType,
    ComponentType,
    Dimension,
    NumberOfComponents,
    imagesize );

  if(retgip != 0)
  {
    return EXIT_FAILURE;
  }

  // Let the user overrule this
  if (NumberOfComponents > 1)
  {
    std::cerr << "ERROR: The NumberOfComponents is larger than 1!" << std::endl;
    std::cerr << "Vector images are not supported!" << std::endl;
    return EXIT_FAILURE;
  }

  // Get rid of the possible "_" in ComponentType.
  ReplaceUnderscoreWithSpace( ComponentType );

  // Run the program.
  bool supported = false;
  int result = EXIT_SUCCESS;
  try
  {
    // 2D
    //run( ProcessImage, char, float, 2 );
    //run( ProcessImage, unsigned char, float, 2 );
    run( ProcessImage, short, float, 2 );
    //run( ProcessImage, float, float, 2 );

    // 3D
    //run( ProcessImage, char, float, 3 );
    //run( ProcessImage, unsigned char, float, 3 );
    run( ProcessImage, short, float, 3 );
    //run( ProcessImage, float, float, 3 );
  }
  catch( itk::ExceptionObject &e )
  {
    std::cerr << "Caught ITK exception: " << e << std::endl;
    result = EXIT_FAILURE;
  }
  if ( !supported )
  {
    std::cerr << "ERROR: this combination of pixeltype and dimension is not supported!" << std::endl;
    std::cerr
      << "pixel (component) type = " << ComponentType
      << " ; dimension = " << Dimension
      << std::endl;
    result = EXIT_FAILURE;
  }

  // End program.
  return result;
}

//------------------------------------------------------------------------------
template<class InputImageType, class OutputImageType>
int ProcessImage(const Parameters &_parameters)
{
  // Typedefs.
  const unsigned int ImageDim = (unsigned int)InputImageType::ImageDimension;
  typedef typename InputImageType::PixelType   InputPixelType;
  typedef typename OutputImageType::PixelType  OutputPixelType;

  // CPU Typedefs
  typedef itk::SmoothingRecursiveGaussianImageFilter<InputImageType, OutputImageType> CPUFilterType;
  typedef itk::ImageFileReader<InputImageType>  CPUReaderType;
  typedef itk::ImageFileWriter<OutputImageType> CPUWriterType;

  // GPU Typedefs
  typedef itk::SmoothingRecursiveGaussianImageFilter<InputImageType, OutputImageType> GPUFilterType;
  typedef itk::ImageFileReader<InputImageType>  GPUReaderType;
  typedef itk::ImageFileWriter<OutputImageType> GPUWriterType;

  // Define sigma array
  typename CPUFilterType::SigmaArrayType sigmaArray;
  for(unsigned int i=0; i<InputImageType::ImageDimension; i++)
  {
    sigmaArray[i] = _parameters.sigma;
  }

  // Reader
  typename CPUReaderType::Pointer CPUReader = CPUReaderType::New();
  CPUReader->SetFileName(_parameters.inputFileName);
  CPUReader->Update();

  // Test CPU
  typename CPUFilterType::Pointer CPUFilter = CPUFilterType::New();

  typename InputImageType::ConstPointer  inputImage  = CPUReader->GetOutput();
  typename InputImageType::RegionType    inputRegion = inputImage->GetBufferedRegion();
  typename InputImageType::SizeType      inputSize   = inputRegion.GetSize();

  // Speed CPU vs GPU
  std::cout << "Testing "<< itk::MultiThreader::GetGlobalMaximumNumberOfThreads() <<" threads for CPU vs GPU:\n";
  std::cout << "Filter type: "<< CPUFilter->GetNameOfClass() <<"\n";

  itk::TimeProbe cputimer;
  cputimer.Start();

  CPUFilter->SetNumberOfThreads( itk::MultiThreader::GetGlobalMaximumNumberOfThreads() );

  if(!_parameters.skipCPU)
  {
    for(unsigned int i=0; i<_parameters.runTimes; i++)
    {
      CPUFilter->SetInput( CPUReader->GetOutput() );
      CPUFilter->SetSigmaArray(sigmaArray);
      CPUFilter->Update();

      if(_parameters.runTimes > 1)
      {
        CPUFilter->Modified();
      }
    }
  }

  cputimer.Stop();

  if(!_parameters.skipCPU)
  {
    std::cout << "CPU " << CPUFilter->GetNameOfClass() << " took " << cputimer.GetMeanTime() << " seconds with "
      << CPUFilter->GetNumberOfThreads() << " threads. run times " << _parameters.runTimes << std::endl;
  }

  if(_parameters.GPUEnable)
  {
    // register object factory for GPU image and filter
    itk::ObjectFactoryBase::RegisterFactory( itk::GPUImageFactory::New() );
    itk::ObjectFactoryBase::RegisterFactory( itk::GPURecursiveGaussianImageFilterFactory::New() );
    itk::ObjectFactoryBase::RegisterFactory( itk::GPUCastImageFilterFactory::New() );
  }

  // Test GPU
  typename GPUReaderType::Pointer GPUReader = GPUReaderType::New();
  GPUReader->SetFileName(_parameters.inputFileName);
  GPUReader->Update();

  typename GPUFilterType::Pointer GPUFilter = GPUFilterType::New();

  bool updateException = false;
  itk::TimeProbe gputimer;
  gputimer.Start();

  if(!_parameters.skipGPU)
  {
    for(unsigned int i=0; i<_parameters.runTimes; i++)
    {
      GPUFilter->SetInput( GPUReader->GetOutput() );
      GPUFilter->SetSigmaArray(sigmaArray);

      try
      {
        GPUFilter->Update();
      }
      catch( itk::ExceptionObject &e )
      {
        std::cerr << "Caught ITK exception during GPUFilter->Update(): " << e << std::endl;
        updateException = updateException || true;
      }

      if(_parameters.runTimes > 1)
      {
        GPUFilter->Modified();
      }
    }
  }

  // GPU buffer has not been copied yet, so we have to make manual update
  itk::GPUExplicitSync<GPUFilterType, OutputImageType>( GPUFilter, false );

  gputimer.Stop();

  if(!_parameters.skipGPU)
  {  
    std::cout << "GPU " << GPUFilter->GetNameOfClass() << " took ";
    if(!updateException)
      std::cout << gputimer.GetMeanTime() << " seconds. run times " << _parameters.runTimes << std::endl;
    else
      std::cout << "<na>. run times " << _parameters.runTimes << std::endl;
  }

  // RMS Error check
  const double epsilon = 0.01;
  float diff = 0.0;
  unsigned int nPix = 0;
  if(!_parameters.skipCPU && !_parameters.skipGPU && !updateException)
  {
    itk::ImageRegionIterator<OutputImageType> cit(CPUFilter->GetOutput(),
      CPUFilter->GetOutput()->GetLargestPossibleRegion());
    itk::ImageRegionIterator<OutputImageType> git(GPUFilter->GetOutput(),
      GPUFilter->GetOutput()->GetLargestPossibleRegion());
    for(cit.GoToBegin(), git.GoToBegin(); !cit.IsAtEnd(); ++cit, ++git)
    {
      float err = vnl_math_abs((float)(cit.Get()) - (float)(git.Get()));
      //if(err > epsilon)
      //  std::cout << "CPU : " << (double)(cit.Get()) << ", GPU : " << (double)(git.Get()) << std::endl;
      diff += err*err;
      nPix++;
    }
  }

  float RMSError = 0.0;
  if(!_parameters.skipCPU && !_parameters.skipGPU && !updateException)
  {
    RMSError = sqrt( diff / (float)nPix );
    std::cout << "RMS Error: " << std::fixed << std::setprecision(8) << RMSError << std::endl;
  }
  bool testPassed = false;
  if (!updateException)
    testPassed = (RMSError <= _parameters.RMSError);

  // Write output
  if(_parameters.outputWrite)
  {
    if(!_parameters.skipCPU)
    {
      // Write output CPU image
      typename CPUWriterType::Pointer writerCPU = CPUWriterType::New();
      writerCPU->SetInput(CPUFilter->GetOutput());
      writerCPU->SetFileName( _parameters.outputFileNames[0] );
      writerCPU->Update();
    }

    if(!_parameters.skipGPU && !updateException)
    {
      // Write output GPU image
      typename GPUWriterType::Pointer writerGPU = GPUWriterType::New();
      writerGPU->SetInput(GPUFilter->GetOutput());
      writerGPU->SetFileName( _parameters.outputFileNames[1] );
      writerGPU->Update();
    }
  }

  // Write log
  if(_parameters.outputLog)
  {
    std::string comments;
    if(updateException)
      comments.append(", Exception during update");

    itk::WriteLog<InputImageType>(
      _parameters.logFileName, ImageDim, inputSize, RMSError,
      testPassed, updateException,
      CPUFilter->GetNumberOfThreads(), _parameters.runTimes,
      CPUFilter->GetNameOfClass(),
      cputimer.GetMeanTime(), gputimer.GetMeanTime(), comments);
  }

  if(testPassed)
    return EXIT_SUCCESS;
  else
    return EXIT_FAILURE;
}

//------------------------------------------------------------------------------
void PrintHelp( void )
{
  std::cout << "Usage:" << std::endl;
  std::cout << "  -in           input file name" << std::endl;
  std::cout << "  [-sigma]      sigma" << std::endl;
  std::cout << "  [-out]        output file names.(outputCPU outputGPU)" << std::endl;
  std::cout << "  [-outlog]     output log file name, default 'CPUGPULog.txt'" << std::endl;
  std::cout << "  [-nooutput]   controls where output is created, default write output" << std::endl;
  std::cout << "  [-runtimes]   controls how many times filter will execute, default 1" << std::endl;
  std::cout << "  [-skipcpu]    skip running CPU part, default false" << std::endl;
  std::cout << "  [-skipgpu]    skip running GPU part, default false" << std::endl;
  std::cout << "  [-rms]        rms error, default 0" << std::endl;
  std::cout << "  [-gpu]        use GPU, default 0" << std::endl;
  std::cout << "  [-threads]    number of threads, default maximum" << std::endl;
}
