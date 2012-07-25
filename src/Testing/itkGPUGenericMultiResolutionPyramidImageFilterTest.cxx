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

// GPU include files
#include "itkGPURecursiveGaussianImageFilter.h"
#include "itkGPUCastImageFilter.h"
#include "itkGPUResampleImageFilter.h"
#include "itkGPUIdentityTransform.h"
#include "itkGPULinearInterpolateImageFunction.h"
#include "itkGPUShrinkImageFilter.h"

#pragma warning(push)
// warning C4267: 'initializing' : conversion from 'size_t' to 'unsigned int',
// possible loss of data
#pragma warning(disable:4267)
#include "itkGenericMultiResolutionPyramidImageFilter.h"
#pragma warning(pop)

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
      outputIndex = 0;

      GPUEnable = 0;
      RMSError = 0.0;
      runTimes = 1;
      skipCPU = false;
      skipGPU = false;

      // Filter
      numlevels = 4;
      currentLevel = 0;
      useMultiResolutionRescaleSchedule = false;
      useMultiResolutionSmoothingSchedule = false;
      useShrinkImageFilter = false;
      useComputeOnlyForCurrentLevel = false;


      // Files
      logFileName = "CPUGPULog.txt";
    }

    bool useCompression;
    bool outputWrite;
    bool outputLog;
    int GPUEnable;
    bool skipCPU;
    bool skipGPU;
    float RMSError;
    unsigned int runTimes;

    // Files
    std::string inputFileName;
    std::vector<std::string> outputFileNames;
    unsigned int outputIndex;
    std::string logFileName;

    // Filter
    unsigned int numlevels;
    unsigned int currentLevel;
    bool useMultiResolutionRescaleSchedule;
    bool useMultiResolutionSmoothingSchedule;
    bool useShrinkImageFilter;
    bool useComputeOnlyForCurrentLevel;
  };
}

//----------------------------------------------------------------------------
template <class GenericPyramidType>
void SetNumberOfLevels(typename GenericPyramidType::Pointer &genericPyramid,
                       const Parameters &_parameters)
{
  const unsigned int numLevels = _parameters.numlevels;
  switch (numLevels)
  {
  case 1:
    {
      itk::Vector<unsigned int, 1> factors;
      unsigned int factor = static_cast<unsigned int>(pow(2.0, 0));
      for(unsigned int i=0; i<1; i++)
      {
        factors[i] = factor;
        factor /= 2;
      }
      genericPyramid->SetNumberOfLevels(1);
      genericPyramid->SetStartingShrinkFactors(factors.Begin());
    }
    break;
  case 2:
    {
      itk::Vector<unsigned int, 2> factors;
      unsigned int factor = static_cast<unsigned int>(pow(2.0, 1));
      for(unsigned int i=0; i<2; i++)
      {
        factors[i] = factor;
        factor /= 2;
      }
      genericPyramid->SetNumberOfLevels(2);
      genericPyramid->SetStartingShrinkFactors(factors.Begin());
    }
    break;
  case 3:
    {
      itk::Vector<unsigned int, 3> factors;
      unsigned int factor = static_cast<unsigned int>(pow(2.0, 2));
      for(unsigned int i=0; i<3; i++)
      {
        factors[i] = factor;
        factor /= 2;
      }
      genericPyramid->SetNumberOfLevels(3);
      genericPyramid->SetStartingShrinkFactors(factors.Begin());
    }
    break;
  case 4:
    {
      itk::Vector<unsigned int, 4> factors;
      unsigned int factor = static_cast<unsigned int>(pow(2.0, 3));
      for(unsigned int i=0; i<4; i++)
      {
        factors[i] = factor;
        factor /= 2;
      }
      genericPyramid->SetNumberOfLevels(4);
      genericPyramid->SetStartingShrinkFactors(factors.Begin());
    }
    break;
  case 5:
    {
      itk::Vector<unsigned int, 5> factors;
      unsigned int factor = static_cast<unsigned int>(pow(2.0, 4));
      for(unsigned int i=0; i<5; i++)
      {
        factors[i] = factor;
        factor /= 2;
      }
      genericPyramid->SetNumberOfLevels(5);
      genericPyramid->SetStartingShrinkFactors(factors.Begin());
    }
    break;
  default:
    std::cerr << "Not implemented." << std::endl;
    break;
  }
}

//------------------------------------------------------------------------------
template<class InputImageType, class OutputImageType>
int ProcessImage(const Parameters &_parameters);

// Testing GPU GenericMultiResolutionPyramid Image Filter

// 2D:
// -in D:\\work\\elastix-ext\\ITK4OpenCL\\data\\image-256x256-2D.mha -out D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test1\\image-256x256-2D-out-cpu.mha D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test1\\image-256x256-2D-out-gpu.mha -rsm 0.0 -gpu
// -in D:\\work\\elastix-ext\\ITK4OpenCL\\data\\image-256x256-2D.mha -out D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test1\\image-256x256-2D-out-cpu.mha D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test1\\image-256x256-2D-out-gpu.mha -mrss -rsm 0.02 -gpu
// -in D:\\work\\elastix-ext\\ITK4OpenCL\\data\\image-256x256-2D.mha -out D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test1\\image-256x256-2D-out-cpu.mha D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test1\\image-256x256-2D-out-gpu.mha -mrrs -rsm 0.0 -gpu
// -in D:\\work\\elastix-ext\\ITK4OpenCL\\data\\image-256x256-2D.mha -out D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test1\\image-256x256-2D-out-cpu.mha D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test1\\image-256x256-2D-out-gpu.mha -mrss -mrrs -rsm 0.001 -gpu
// -in D:\\work\\elastix-ext\\ITK4OpenCL\\data\\image-256x256-2D.mha -out D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test1\\image-256x256-2D-out-cpu.mha D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test1\\image-256x256-2D-out-gpu.mha -mrss -sif -rsm 0.02 -gpu
// -in D:\\work\\elastix-ext\\ITK4OpenCL\\data\\image-256x256-2D.mha -out D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test1\\image-256x256-2D-out-cpu.mha D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test1\\image-256x256-2D-out-gpu.mha -mrrs -sif -rsm 0.0 -gpu
// -in D:\\work\\elastix-ext\\ITK4OpenCL\\data\\image-256x256-2D.mha -out D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test1\\image-256x256-2D-out-cpu.mha D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test1\\image-256x256-2D-out-gpu.mha -mrss -mrrs -sif -rsm 0.001 -gpu

// 2D with -mrrs -outindex 1:
// -in D:\\work\\elastix-ext\\ITK4OpenCL\\data\\image-256x256-2D.mha -out D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test1\\image-256x256-2D-out-cpu.mha D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test1\\image-256x256-2D-out-gpu.mha -outindex 1 -mrss -mrrs -rsm 0.001 -gpu

// 3D:
// -in D:\\work\\elastix-ext\\ITK4OpenCL\\data\\image-256x256x256-3D.mha -out D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test1\\image-256x256x256-3D-out-cpu.mha D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test1\\image-256x256x256-3D-out-gpu.mha -rsm 0.001 -gpu
// -in D:\\work\\elastix-ext\\ITK4OpenCL\\data\\image-256x256x256-3D.mha -out D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test1\\image-256x256x256-3D-out-cpu.mha D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test1\\image-256x256x256-3D-out-gpu.mha -mrss -rsm 0.3 -gpu
// -in D:\\work\\elastix-ext\\ITK4OpenCL\\data\\image-256x256x256-3D.mha -out D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test1\\image-256x256x256-3D-out-cpu.mha D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test1\\image-256x256x256-3D-out-gpu.mha -mrrs -rsm 0.2 -gpu
// -in D:\\work\\elastix-ext\\ITK4OpenCL\\data\\image-256x256x256-3D.mha -out D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test1\\image-256x256x256-3D-out-cpu.mha D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test1\\image-256x256x256-3D-out-gpu.mha -mrss -mrrs -rsm 0.01 -gpu
// -in D:\\work\\elastix-ext\\ITK4OpenCL\\data\\image-256x256x256-3D.mha -out D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test1\\image-256x256x256-3D-out-cpu.mha D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test1\\image-256x256x256-3D-out-gpu.mha -mrss -sif -rsm 0.3 -gpu
// -in D:\\work\\elastix-ext\\ITK4OpenCL\\data\\image-256x256x256-3D.mha -out D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test1\\image-256x256x256-3D-out-cpu.mha D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test1\\image-256x256x256-3D-out-gpu.mha -mrrs -sif -rsm 0.001 -gpu
// -in D:\\work\\elastix-ext\\ITK4OpenCL\\data\\image-256x256x256-3D.mha -out D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test1\\image-256x256x256-3D-out-cpu.mha D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test1\\image-256x256x256-3D-out-gpu.mha -mrss -mrrs -sif -rsm 0.01 -gpu

// 3D BIG:
// -in D:\\work\\elastix-ext\\ITK4OpenCL\\data\\image-512x512x256-3D.mha -out D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test1\\image-512x512x256-3D-out-cpu.mha D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test1\\image-512x512x256-3D-out-gpu.mha -rsm 0.001 -gpu
// -in D:\\work\\elastix-ext\\ITK4OpenCL\\data\\image-512x512x256-3D.mha -out D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test1\\image-512x512x256-3D-out-cpu.mha D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test1\\image-512x512x256-3D-out-gpu.mha -mrss -rsm 0.1 -gpu
// -in D:\\work\\elastix-ext\\ITK4OpenCL\\data\\image-512x512x256-3D.mha -out D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test1\\image-512x512x256-3D-out-cpu.mha D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test1\\image-512x512x256-3D-out-gpu.mha -mrrs -rsm 0.007 -gpu
// -in D:\\work\\elastix-ext\\ITK4OpenCL\\data\\image-512x512x256-3D.mha -out D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test1\\image-512x512x256-3D-out-cpu.mha D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test1\\image-512x512x256-3D-out-gpu.mha -mrss -mrrs -rsm 0.01 -gpu
// -in D:\\work\\elastix-ext\\ITK4OpenCL\\data\\image-512x512x256-3D.mha -out D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test1\\image-512x512x256-3D-out-cpu.mha D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test1\\image-512x512x256-3D-out-gpu.mha -mrss -sif -rsm 0.0003 -gpu
// -in D:\\work\\elastix-ext\\ITK4OpenCL\\data\\image-512x512x256-3D.mha -out D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test1\\image-512x512x256-3D-out-cpu.mha D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test1\\image-512x512x256-3D-out-gpu.mha -mrrs -sif -rsm 0.0005 -gpu
// -in D:\\work\\elastix-ext\\ITK4OpenCL\\data\\image-512x512x256-3D.mha -out D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test1\\image-512x512x256-3D-out-cpu.mha D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test1\\image-512x512x256-3D-out-gpu.mha -mrss -mrrs -sif -rsm 0.0005 -gpu

// 3D with -mrrs -outindex 1:
// -in D:\\work\\elastix-ext\\ITK4OpenCL\\data\\image-256x256x256-3D.mha -out D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test1\\image-256x256x256-3D-out-cpu.mha D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test1\\image-256x256x256-3D-out-gpu.mha -outindex 1 -mrss -mrrs -rsm 0.001 -gpu
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
  parser->GetCommandLineArgument("-outindex", parameters.outputIndex);
  parser->GetCommandLineArgument("-outlog", parameters.logFileName);
  parser->GetCommandLineArgument("-runtimes", parameters.runTimes);

  parser->GetCommandLineArgument("-rsm", parameters.RMSError);
  parameters.GPUEnable = parser->ArgumentExists("-gpu");
  parameters.skipCPU = parser->ArgumentExists("-skipcpu");
  parameters.skipGPU = parser->ArgumentExists("-skipgpu");

  // Filter parameters
  parser->GetCommandLineArgument("-numlevels", parameters.numlevels);
  parameters.useMultiResolutionRescaleSchedule = parser->ArgumentExists("-mrrs");
  parameters.useMultiResolutionSmoothingSchedule = parser->ArgumentExists("-mrss");
  parameters.useShrinkImageFilter = parser->ArgumentExists("-sif");
  parameters.useComputeOnlyForCurrentLevel = parser->ArgumentExists("-cl");
  parser->GetCommandLineArgument("-l", parameters.currentLevel);

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
  typedef itk::GenericMultiResolutionPyramidImageFilter<InputImageType, OutputImageType, float> CPUFilterType;
  typedef typename CPUFilterType::ScheduleType CPUScheduleType;
  typedef itk::ImageFileReader<InputImageType>  CPUReaderType;
  typedef itk::ImageFileWriter<OutputImageType> CPUWriterType;

  // GPU Typedefs
  typedef itk::GenericMultiResolutionPyramidImageFilter<InputImageType, OutputImageType, float> GPUFilterType;
  typedef typename GPUFilterType::ScheduleType GPUScheduleType;
  typedef itk::ImageFileReader<InputImageType>  GPUReaderType;
  typedef itk::ImageFileWriter<OutputImageType> GPUWriterType;

  // CPU Reader
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
      SetNumberOfLevels<CPUFilterType>(CPUFilter, _parameters);
      CPUFilter->SetInput( CPUReader->GetOutput() );
      CPUFilter->SetUseMultiResolutionRescaleSchedule(_parameters.useMultiResolutionRescaleSchedule);
      CPUFilter->SetUseMultiResolutionSmoothingSchedule(_parameters.useMultiResolutionSmoothingSchedule);
      CPUFilter->SetUseShrinkImageFilter(_parameters.useShrinkImageFilter);

      if(!_parameters.useComputeOnlyForCurrentLevel)
      {
        CPUFilter->SetComputeOnlyForCurrentLevel(_parameters.useComputeOnlyForCurrentLevel);
        CPUFilter->SetCurrentLevel(_parameters.currentLevel);
        CPUFilter->Update();
      }
      else
      {
        CPUFilter->SetComputeOnlyForCurrentLevel(_parameters.useComputeOnlyForCurrentLevel);
        for(unsigned int i=0; i<CPUFilter->GetNumberOfLevels(); i++)
        {
          CPUFilter->SetCurrentLevel(i);
          CPUFilter->Update();
        }
      }

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

    if(_parameters.useShrinkImageFilter)
    {
      itk::ObjectFactoryBase::RegisterFactory( itk::GPUShrinkImageFilterFactory::New() );
    }
    else
    {
      itk::ObjectFactoryBase::RegisterFactory( itk::GPUResampleImageFilterFactory::New() );
      itk::ObjectFactoryBase::RegisterFactory( itk::GPUIdentityTransformFactory::New() );
      itk::ObjectFactoryBase::RegisterFactory( itk::GPULinearInterpolateImageFunctionFactory::New() );
    }
  }

  // GPU reader
  typename GPUReaderType::Pointer GPUReader = GPUReaderType::New();
  GPUReader->SetFileName(_parameters.inputFileName);
  GPUReader->Update();

  // Test GPU
  typename GPUFilterType::Pointer GPUFilter = GPUFilterType::New();

  bool updateException = false;
  itk::TimeProbe gputimer;
  gputimer.Start();

  if(!_parameters.skipGPU)
  {
    for(unsigned int i=0; i<_parameters.runTimes; i++)
    {
      SetNumberOfLevels<GPUFilterType>(GPUFilter, _parameters);
      GPUFilter->SetInput( GPUReader->GetOutput() );
      GPUFilter->SetUseMultiResolutionRescaleSchedule(_parameters.useMultiResolutionRescaleSchedule);
      GPUFilter->SetUseMultiResolutionSmoothingSchedule(_parameters.useMultiResolutionSmoothingSchedule);
      GPUFilter->SetUseShrinkImageFilter(_parameters.useShrinkImageFilter);
      GPUFilter->SetComputeOnlyForCurrentLevel(_parameters.useComputeOnlyForCurrentLevel);
      GPUFilter->SetCurrentLevel(_parameters.currentLevel);

      try
      {
        if(!_parameters.useComputeOnlyForCurrentLevel)
        {
          GPUFilter->SetComputeOnlyForCurrentLevel(_parameters.useComputeOnlyForCurrentLevel);
          GPUFilter->SetCurrentLevel(_parameters.currentLevel);
          GPUFilter->Update();
        }
        else
        {
          GPUFilter->SetComputeOnlyForCurrentLevel(_parameters.useComputeOnlyForCurrentLevel);
          for(unsigned int i=0; i<CPUFilter->GetNumberOfLevels(); i++)
          {
            GPUFilter->SetCurrentLevel(i);
            GPUFilter->Update();
          }
        }
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
    itk::ImageRegionIterator<OutputImageType> cit(CPUFilter->GetOutput(_parameters.outputIndex),
      CPUFilter->GetOutput(_parameters.outputIndex)->GetLargestPossibleRegion());
    itk::ImageRegionIterator<OutputImageType> git(GPUFilter->GetOutput(_parameters.outputIndex),
      GPUFilter->GetOutput(_parameters.outputIndex)->GetLargestPossibleRegion());
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
      writerCPU->SetInput(CPUFilter->GetOutput(_parameters.outputIndex));
      writerCPU->SetFileName( _parameters.outputFileNames[0] );
      writerCPU->Update();
    }

    if(!_parameters.skipGPU && !updateException)
    {
      // Write output GPU image
      typename GPUWriterType::Pointer writerGPU = GPUWriterType::New();
      writerGPU->SetInput(GPUFilter->GetOutput(_parameters.outputIndex));
      writerGPU->SetFileName( _parameters.outputFileNames[1] );
      writerGPU->Update();
    }
  }

  // Write log
  if(_parameters.outputLog)
  {
    std::string comments;

    // Add RescaleSchedule comments
    if(_parameters.useMultiResolutionRescaleSchedule)
      comments.append("RescaleSchedule(On)");
    else
      comments.append("RescaleSchedule(Off)");

    comments.append(", ");

    // Add SmoothingSchedule comments
    if(_parameters.useMultiResolutionSmoothingSchedule)
      comments.append("SmoothingSchedule(On)");
    else
      comments.append("SmoothingSchedule(Off)");

    comments.append(", ");

    // Add ShrinkImageFilter comments
    if(_parameters.useShrinkImageFilter)
      comments.append("ShrinkImageFilter(On)");
    else
      comments.append("ShrinkImageFilter(Off)");

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
  std::cout << "  [-outindex]   output index" << std::endl;
  std::cout << "  [-outlog]     output log file name, default 'CPUGPULog.txt'" << std::endl;
  std::cout << "  [-nooutput]   controls where output is created, default write output" << std::endl;
  std::cout << "  [-runtimes]   controls how many times filter will execute, default 1" << std::endl;
  std::cout << "  [-skipcpu]    skip running CPU part, default false" << std::endl;
  std::cout << "  [-skipgpu]    skip running GPU part, default false" << std::endl;
  std::cout << "  [-numlevels]  number of levels" << std::endl;
  std::cout << "  [-mrrs]       use multi resolution rescale schedule, default false" << std::endl;
  std::cout << "  [-mrss]       use multi resolution smoothing schedule, default false" << std::endl;
  std::cout << "  [-sif]        use shrink image filter, default false" << std::endl;
  std::cout << "  [-cl]         use current level, default false" << std::endl;
  std::cout << "  [-l]          current level, default 0" << std::endl;
  std::cout << "  [-out]        output file names.(outputCPU outputGPU)" << std::endl;
  std::cout << "  [-rms]        rms error, default 0" << std::endl;
  std::cout << "  [-gpu]        use GPU, default 0" << std::endl;
  std::cout << "  [-threads]    number of threads, default maximum" << std::endl;
}
