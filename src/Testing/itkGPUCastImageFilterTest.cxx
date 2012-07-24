/*=========================================================================
*
*  Copyright Insight Software Consortium
*
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*         http://www.apache.org/licenses/LICENSE-2.0.txt
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*
*=========================================================================*/
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

// GPU include files
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
  };
}

//------------------------------------------------------------------------------
template<class InputImageType, class OutputImageType>
int ProcessImage(const Parameters &_parameters);

// Testing GPU BSplineDecompositionImageFilter
// 1D:
// -in D:\\work\\elastix-ext\\ITK4OpenCL\\data\\image-512-1D.mha -out D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test5\\image-512-1D-out-cpu.mha D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test5\\image-512-1D-out-gpu.mha -gpu -rsm 0.0
// 2D:
// -in D:\\work\\elastix-ext\\ITK4OpenCL\\data\\image-256x256-2D.mha -out D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test5\\image-256x256-2D-out-cpu.mha D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test5\\image-256x256-2D-out-gpu.mha -gpu -rsm 0.0
// -in D:\\work\\elastix-ext\\ITK4OpenCL\\data\\image-coefficients-2D.mha -out D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test5\\image-coefficients-2D-out-cpu.mha D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test5\\image-coefficients-2D-out-gpu.mha -gpu -rsm 0.0
// 3D:
// -in D:\\work\\elastix-ext\\ITK4OpenCL\\data\\image-64x64x64-3D.mha -out D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test5\\image-64x64x64-3D-out-cpu.mha D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test5\\image-64x64x64-3D-out-gpu.mha -gpu -rsm 0.0'

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
    // 1D
    //run( ProcessImage, char, float, 1 );
    //run( ProcessImage, unsigned char, float, 1 );
    run( ProcessImage, short, float, 1 );
    //run( ProcessImage, float, float, 1 );

    // 2D
    //run( ProcessImage, char, float, 2 );
    //run( ProcessImage, unsigned char, float, 2 );
    run( ProcessImage, short, float, 2 );
    run( ProcessImage, double, float, 2 );

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
  typedef itk::Image<double, InputImageType::ImageDimension> RealInputImageType;
  const unsigned int ImageDim = (unsigned int)InputImageType::ImageDimension;
  typedef typename InputImageType::PixelType  InputPixelType;
  typedef typename OutputImageType::PixelType OutputPixelType;

  typedef itk::GPUImage<OutputPixelType, OutputImageType::ImageDimension> GPUOutputImageType;

  // CPU Typedefs
  typedef itk::CastImageFilter<RealInputImageType, OutputImageType> CPUFilterType;
  typedef itk::ImageFileReader<RealInputImageType> CPUReaderType;
  typedef itk::ImageFileWriter<OutputImageType> CPUWriterType;
  typedef typename OutputImageType::Pointer CPUOutputImageImagePointer;

  // GPU Typedefs
  typedef itk::CastImageFilter<RealInputImageType, GPUOutputImageType> GPUFilterType;
  typedef itk::ImageFileReader<RealInputImageType> GPUReaderType;
  typedef itk::ImageFileWriter<OutputImageType> GPUWriterType;

  typedef typename GPUOutputImageType::Pointer GPUOutputImageImagePointer;

  // Reader
  typename CPUReaderType::Pointer CPUReader = CPUReaderType::New();
  CPUReader->SetFileName(_parameters.inputFileName);
  CPUReader->Update();

  // Test CPU
  typename CPUFilterType::Pointer CPUFilter = CPUFilterType::New();

  typename RealInputImageType::ConstPointer inputImage  = CPUReader->GetOutput();
  typename RealInputImageType::RegionType   inputRegion = inputImage->GetBufferedRegion();
  typename RealInputImageType::SizeType     inputSize   = inputRegion.GetSize();

  // Speed CPU vs GPU
  std::cout << "Testing "<< itk::MultiThreader::GetGlobalMaximumNumberOfThreads() <<" threads for CPU vs GPU:\n";
  std::cout << "Filter type: "<< CPUFilter->GetNameOfClass() <<"\n";

  CPUOutputImageImagePointer cpuOutputImage = OutputImageType::New();
  cpuOutputImage->CopyInformation( CPUReader->GetOutput() );
  cpuOutputImage->SetRegions( CPUReader->GetOutput()->GetBufferedRegion() );
  cpuOutputImage->Allocate();

  CPUFilter->SetNumberOfThreads( itk::MultiThreader::GetGlobalMaximumNumberOfThreads() );

  itk::TimeProbe cputimer;
  cputimer.Start();

  if(!_parameters.skipCPU)
  {
    for(unsigned int i=0; i<_parameters.runTimes; i++)
    {
      CPUFilter->SetInput( CPUReader->GetOutput() );
      CPUFilter->GraftOutput( cpuOutputImage );
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
    itk::ObjectFactoryBase::RegisterFactory( itk::GPUCastImageFilterFactory::New() );
  }

  typename GPUReaderType::Pointer GPUReader = GPUReaderType::New();
  GPUReader->SetFileName(_parameters.inputFileName);
  GPUReader->Update();

  // Test GPU
  typename GPUFilterType::Pointer GPUFilter = GPUFilterType::New();

  GPUOutputImageImagePointer gpuOutputImage = GPUOutputImageType::New();
  gpuOutputImage->CopyInformation( GPUReader->GetOutput() );
  gpuOutputImage->SetRegions( GPUReader->GetOutput()->GetBufferedRegion() );
  gpuOutputImage->Allocate();

  bool updateException = false;
  itk::TimeProbe gputimer;
  gputimer.Start();

  if(!_parameters.skipGPU)
  {
    for(unsigned int i=0; i<_parameters.runTimes; i++)
    {
      GPUFilter->SetInput( GPUReader->GetOutput() );
      GPUFilter->GraftOutput( gpuOutputImage );

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
  if (!updateException)
    itk::GPUExplicitSync<GPUFilterType, GPUOutputImageType>( GPUFilter, false );

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
    itk::ImageRegionIterator<OutputImageType> cit(cpuOutputImage,
      cpuOutputImage->GetLargestPossibleRegion());
    itk::ImageRegionIterator<GPUOutputImageType> git(gpuOutputImage,
      gpuOutputImage->GetLargestPossibleRegion());

    for(cit.GoToBegin(), git.GoToBegin(); !cit.IsAtEnd(); ++cit, ++git)
    {
      float c = (float)(cit.Get());
      float g = (float)(git.Get());
      float err = vnl_math_abs( c - g );
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
