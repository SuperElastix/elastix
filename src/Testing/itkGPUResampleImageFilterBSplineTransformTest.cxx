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
#include "itkMinimumMaximumImageCalculator.h"
#include "itkBinaryThresholdImageFilter.h"

// GPU include files
#pragma warning(push)
// warning C4355: 'this' : used in base member initializer list
#pragma warning(disable:4355)
#include "itkGPUResampleImageFilter.h"
#include "itkGPUBSplineTransform.h"

#include "itkGPUNearestNeighborInterpolateImageFunction.h"
#include "itkGPULinearInterpolateImageFunction.h"
#include "itkGPUBSplineInterpolateImageFunction.h"
#include "itkGPUBSplineDecompositionImageFilter.h"
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
      GPUEnable = 0;
      RMSError = 0.0;
      runTimes = 1;
      skipCPU = false;
      skipGPU = false;
      interpolator = "NearestNeighbor";

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
    std::string interpolator;

    // Files
    std::vector<std::string> inputFileNames;
    std::string inputParametersFileName;
    std::vector<std::string> outputFileNames;
    std::string logFileName;

    // Filter
  };
}

//------------------------------------------------------------------------------
template<class InputImageType, class OutputImageType>
int ProcessImage(const Parameters &_parameters);

// Testing GPU Rescale Image Filter with BSplineTransform

// Interpolator NearestNeighbor:
// 1D:
// -in D:\\work\\elastix-ext\\ITK4OpenCL\\data\\image-512-1D.mha -inpar D:\\work\\elastix-ext\\ITK4OpenCL\\data\\BSplineDisplacements2-3D.txt -out D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test11\\image-512-1D-out-cpu.mha D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test11\\image-512-1D-out-gpu.mha -gpu -rsm 0.0
// 2D:
// -in D:\\work\\elastix-ext\\ITK4OpenCL\\data\\image-256x256-2D.mha -inpar D:\\work\\elastix-ext\\ITK4OpenCL\\data\\BSplineDisplacements5-2D.txt -out D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test11\\image-256x256-2D-out-cpu.mha D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test11\\image-256x256-2D-out-gpu.mha -gpu -rsm 0.0
// 3D:
// -in D:\\work\\elastix-ext\\ITK4OpenCL\\data\\image-256x256x256-3D.mha -inpar D:\\work\\elastix-ext\\ITK4OpenCL\\data\\BSplineDisplacements2-3D.txt -out D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test11\\image-256x256x256-3D-out-cpu.mha D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test11\\image-256x256x256-3D-out-gpu.mha -gpu -rsm 0.26
// 3D:
// -in D:\\work\\elastix-ext\\ITK4OpenCL\\data\\image-64x64x64-3D.mha -inpar D:\\work\\elastix-ext\\ITK4OpenCL\\data\\BSplineDisplacements2-3D.txt -out D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test11\\image-64x64x64-3D-out-cpu.mha D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test11\\image-64x64x64-3D-out-gpu.mha -gpu -rsm 0.0

// Interpolator Linear:
// 1D:
// -in D:\\work\\elastix-ext\\ITK4OpenCL\\data\\image-512-1D.mha -inpar D:\\work\\elastix-ext\\ITK4OpenCL\\data\\BSplineDisplacements2-3D.txt -out D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test12\\image-512-1D-out-cpu.mha D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test12\\image-512-1D-out-gpu.mha -i Linear -gpu -rsm 0.0
// 2D:
// -in D:\\work\\elastix-ext\\ITK4OpenCL\\data\\image-256x256-2D.mha -inpar D:\\work\\elastix-ext\\ITK4OpenCL\\data\\BSplineDisplacements5-2D.txt -out D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test12\\image-256x256-2D-out-cpu.mha D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test12\\image-256x256-2D-out-gpu.mha -i Linear -gpu -rsm 0.02
// 3D:
// -in D:\\work\\elastix-ext\\ITK4OpenCL\\data\\image-256x256x256-3D.mha -inpar D:\\work\\elastix-ext\\ITK4OpenCL\\data\\BSplineDisplacements2-3D.txt -out D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test12\\image-256x256x256-3D-out-cpu.mha D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test12\\image-256x256x256-3D-out-gpu.mha -i Linear -gpu -rsm 0.6
// 3D:
// -in D:\\work\\elastix-ext\\ITK4OpenCL\\data\\image-64x64x64-3D.mha -inpar D:\\work\\elastix-ext\\ITK4OpenCL\\data\\BSplineDisplacements2-3D.txt -out D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test12\\image-64x64x64-3D-out-cpu.mha D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test12\\image-64x64x64-3D-out-gpu.mha -i Linear -gpu -rsm 5.0

// Interpolator BSpline:
// 1D:
// -in D:\\work\\elastix-ext\\ITK4OpenCL\\data\\image-512-1D.mha -inpar D:\\work\\elastix-ext\\ITK4OpenCL\\data\\BSplineDisplacements2-3D.txt -out D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test13\\image-512-1D-out-cpu.mha D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test13\\image-512-1D-out-gpu.mha -i BSpline -gpu -rsm 0.0
// 2D:
// -in D:\\work\\elastix-ext\\ITK4OpenCL\\data\\image-256x256-2D.mha -inpar D:\\work\\elastix-ext\\ITK4OpenCL\\data\\BSplineDisplacements5-2D.txt -out D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test13\\image-256x256-2D-out-cpu.mha D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test13\\image-256x256-2D-out-gpu.mha -i BSpline -gpu -rsm 0.0
// 3D:
// -in D:\\work\\elastix-ext\\ITK4OpenCL\\data\\image-256x256x256-3D.mha -inpar D:\\work\\elastix-ext\\ITK4OpenCL\\data\\BSplineDisplacements2-3D.txt -out D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test13\\image-256x256x256-3D-out-cpu.mha D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test13\\image-256x256x256-3D-out-gpu.mha -i BSpline -gpu -rsm 1.5
// 3D:
// -in D:\\work\\elastix-ext\\ITK4OpenCL\\data\\image-64x64x64-3D.mha -inpar D:\\work\\elastix-ext\\ITK4OpenCL\\data\\BSplineDisplacements2-3D.txt -out D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test13\\image-64x64x64-3D-out-cpu.mha D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test13\\image-64x64x64-3D-out-gpu.mha -i BSpline -gpu -rsm 8.0

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
  const bool retin = parser->GetCommandLineArgument("-in", parameters.inputFileNames);
  const bool retinpar = parser->GetCommandLineArgument("-inpar", parameters.inputParametersFileName);
  parameters.outputFileNames.push_back(parameters.inputFileNames[0].substr(0, parameters.inputFileNames[0].rfind( "." ))+"-out-cpu.mha");
  parameters.outputFileNames.push_back(parameters.inputFileNames[0].substr(0, parameters.inputFileNames[0].rfind( "." ))+"-out-gpu.mha");
  parser->GetCommandLineArgument("-out", parameters.outputFileNames);
  parameters.outputWrite = !(parser->ArgumentExists( "-nooutput" ));
  parser->GetCommandLineArgument("-outlog", parameters.logFileName);
  const bool retruntimes = parser->GetCommandLineArgument("-runtimes", parameters.runTimes);

  parser->GetCommandLineArgument("-rsm", parameters.RMSError);
  parser->GetCommandLineArgument("-i", parameters.interpolator);
  parameters.GPUEnable = parser->ArgumentExists("-gpu");
  parameters.skipCPU = parser->ArgumentExists("-skipcpu");
  parameters.skipGPU = parser->ArgumentExists("-skipgpu");

  // Threads.
  unsigned int maximumNumberOfThreads = itk::MultiThreader::GetGlobalDefaultNumberOfThreads();
  parser->GetCommandLineArgument( "-threads", maximumNumberOfThreads );
  itk::MultiThreader::SetGlobalMaximumNumberOfThreads( maximumNumberOfThreads );

  // Check if the required arguments are given.
  if(!retin)
  {
    std::cerr << "ERROR: You should specify \"-in\"." << std::endl;
    return EXIT_FAILURE;
  }

  if(!retinpar)
  {
    std::cerr << "ERROR: You should specify \"-inpar\"." << std::endl;
    return EXIT_FAILURE;
  }

  if(retruntimes && parameters.runTimes < 1)
  {
    std::cerr << "ERROR: \"-runtimes\" parameter should be more or equal one." << std::endl;
    return EXIT_FAILURE;
  }

  if(parameters.interpolator != "NearestNeighbor"
    && parameters.interpolator != "Linear"
    && parameters.interpolator != "BSpline")
  {
    std::cerr << "ERROR: interpolator \"-i\" should be one of {NearestNeighbor, Linear, BSpline}." << std::endl;
    return EXIT_FAILURE;
  }

  // Determine image properties.
  std::string ComponentType = "short";
  std::string PixelType; //we don't use this
  unsigned int Dimension = 2;
  unsigned int NumberOfComponents = 1;
  std::vector<unsigned int> imagesize( Dimension, 0 );
  int retgip = GetImageProperties(
    parameters.inputFileNames[0],
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
    run( ProcessImage, short, short, 1 );
    //run( ProcessImage, float, float, 1 );

    // 2D
    //run( ProcessImage, char, float, 2 );
    //run( ProcessImage, unsigned char, float, 2 );
    run( ProcessImage, short, short, 2 );
    //run( ProcessImage, float, float, 2 );

    // 3D
    //run( ProcessImage, char, float, 3 );
    //run( ProcessImage, unsigned char, float, 3 );
    run( ProcessImage, short, short, 3 );
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
  const unsigned int SpaceDimension = ImageDim;

  typedef typename InputImageType::PixelType  InputPixelType;
  typedef typename OutputImageType::PixelType OutputPixelType;

  // CPU Typedefs
  typedef itk::ResampleImageFilter<InputImageType, OutputImageType, float> CPUFilterType;
  typedef itk::BSplineTransform<float, SpaceDimension, 3>                  CPUTransformType;

  // CPU interpolator typedefs
  typedef itk::InterpolateImageFunction<InputImageType, float>                CPUInterpolatorType;
  typedef typename CPUInterpolatorType::Pointer                               CPUInterpolatorPointerType;
  typedef itk::NearestNeighborInterpolateImageFunction<InputImageType, float> CPUNearestNeighborInterpolatorType;
  typedef itk::LinearInterpolateImageFunction<InputImageType, float>          CPULinearInterpolatorType;
  typedef itk::BSplineInterpolateImageFunction<InputImageType, float, float>  CPUBSplineInterpolatorType;

  // CPU reader/writer
  typedef itk::ImageFileReader<InputImageType>  CPUReaderType;
  typedef itk::ImageFileWriter<OutputImageType> CPUWriterType;

  // GPU Typedefs
  typedef itk::ResampleImageFilter<InputImageType, OutputImageType, float> GPUFilterType;
  typedef itk::BSplineTransform<float, SpaceDimension, 3>                  GPUTransformType;

  // GPU interpolator typedefs
  typedef itk::InterpolateImageFunction<InputImageType, float>                GPUInterpolatorType;
  typedef typename GPUInterpolatorType::Pointer                               GPUInterpolatorPointerType;
  typedef itk::NearestNeighborInterpolateImageFunction<InputImageType, float> GPUNearestNeighborInterpolatorType;
  typedef itk::LinearInterpolateImageFunction<InputImageType, float>          GPULinearInterpolatorType;
  typedef itk::BSplineInterpolateImageFunction<InputImageType, float, float>  GPUBSplineInterpolatorType;

  // GPU reader/writer
  typedef itk::ImageFileReader<InputImageType>  GPUReaderType;
  typedef itk::ImageFileWriter<OutputImageType> GPUWriterType;

  // Read CPU
  typename CPUReaderType::Pointer CPUReader = CPUReaderType::New();
  CPUReader->SetFileName( _parameters.inputFileNames[0] );
  CPUReader->Update();

  typename InputImageType::ConstPointer  inputImage     = CPUReader->GetOutput();
  typename InputImageType::SpacingType   inputSpacing   = inputImage->GetSpacing();
  typename InputImageType::PointType     inputOrigin    = inputImage->GetOrigin();
  typename InputImageType::DirectionType inputDirection = inputImage->GetDirection();
  typename InputImageType::RegionType    inputRegion    = inputImage->GetBufferedRegion();
  typename InputImageType::SizeType      inputSize      = inputRegion.GetSize();

  // Compute min
  typedef itk::MinimumMaximumImageCalculator<InputImageType> MinimumMaximumImageCalculatorType;
  typename MinimumMaximumImageCalculatorType::Pointer calculator = MinimumMaximumImageCalculatorType::New();
  calculator->SetImage(inputImage);
  calculator->ComputeMinimum();
  const typename InputImageType::PixelType minValue = calculator->GetMinimum();

  typedef typename CPUTransformType::MeshSizeType MeshSizeType;
  MeshSizeType meshSize;
  meshSize.Fill( 4 );

  typedef typename CPUTransformType::PhysicalDimensionsType PhysicalDimensionsType;
  PhysicalDimensionsType fixedDimensions;
  for(unsigned int d=0; d<ImageDim; d++)
  {
    fixedDimensions[d] = inputSpacing[d] * ( inputSize[d] - 1.0 );
  }

  // Create CPUTransform
  typename CPUTransformType::Pointer CPUTransform = CPUTransformType::New();
  CPUTransform->SetTransformDomainOrigin( inputOrigin );
  CPUTransform->SetTransformDomainDirection( inputDirection );
  CPUTransform->SetTransformDomainPhysicalDimensions( fixedDimensions );
  CPUTransform->SetTransformDomainMeshSize( meshSize );

  // Read and set parameters
  typedef typename CPUTransformType::ParametersType ParametersType;
  const unsigned int numberOfParameters = CPUTransform->GetNumberOfParameters();
  ParametersType parameters( numberOfParameters );
  std::ifstream infile;
  infile.open( _parameters.inputParametersFileName.c_str() );

  const unsigned int numberOfNodes = numberOfParameters / SpaceDimension;
  for( unsigned int n=0; n < numberOfNodes; n++ )
  {
    unsigned int parValue;
    infile >> parValue;
    parameters[n] = parValue;
    if(SpaceDimension > 1)
      parameters[n+numberOfNodes] = parValue;
    if(SpaceDimension > 2)
      parameters[n+numberOfNodes*2] = parValue;
  }
  infile.close();

  CPUTransform->SetParameters( parameters );

  // Test CPU
  typename CPUFilterType::Pointer CPUFilter = CPUFilterType::New();

  // Create CPU interpolator here
  CPUInterpolatorPointerType CPUInterpolator;
  if(_parameters.interpolator == "NearestNeighbor")
  {
    typename CPUNearestNeighborInterpolatorType::Pointer CPUNearestNeighborInterpolator
      = CPUNearestNeighborInterpolatorType::New();
    CPUInterpolator = CPUNearestNeighborInterpolator;
  }
  else if(_parameters.interpolator == "Linear")
  {
    typename CPULinearInterpolatorType::Pointer CPULinearInterpolator
      = CPULinearInterpolatorType::New();
    CPUInterpolator = CPULinearInterpolator;
  }
  else if(_parameters.interpolator == "BSpline")
  {
    typename CPUBSplineInterpolatorType::Pointer CPUBSplineInterpolator
      = CPUBSplineInterpolatorType::New();
    CPUInterpolator = CPUBSplineInterpolator;
  }

  // Define default value
  const typename OutputImageType::PixelType defaultValue = minValue - 2;

  // Speed CPU vs GPU
  std::cout << "Testing "<< itk::MultiThreader::GetGlobalMaximumNumberOfThreads() <<" threads for CPU vs GPU:\n";
  std::cout << "Interpolator type: "<< CPUInterpolator->GetNameOfClass() <<"\n";
  std::cout << "Transform type: "<< CPUTransform->GetNameOfClass() <<"\n";

  itk::TimeProbe cputimer;
  cputimer.Start();

  CPUFilter->SetNumberOfThreads( itk::MultiThreader::GetGlobalMaximumNumberOfThreads() );

  if(!_parameters.skipCPU)
  {
    for(unsigned int i=0; i<_parameters.runTimes; i++)
    {
      CPUFilter->SetInput( CPUReader->GetOutput() );
      CPUFilter->SetTransform( CPUTransform );
      CPUFilter->SetInterpolator( CPUInterpolator );
      CPUFilter->SetDefaultPixelValue( defaultValue );
      CPUFilter->SetOutputSpacing( inputSpacing );
      CPUFilter->SetOutputOrigin( inputOrigin );
      CPUFilter->SetOutputDirection( inputDirection );
      CPUFilter->SetSize( inputSize );
      CPUFilter->SetOutputStartIndex( inputRegion.GetIndex() );
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
      << CPUFilter->GetNumberOfThreads() << " threads." << " run times " << _parameters.runTimes << std::endl;
  }

  itk::ObjectFactoryBase::Pointer imageFactory = NULL;
  if(_parameters.GPUEnable)
  {
    // register object factory for GPU image and filter
    imageFactory = itk::GPUImageFactory::New();
    itk::ObjectFactoryBase::RegisterFactory( imageFactory );
    itk::ObjectFactoryBase::RegisterFactory( itk::GPUResampleImageFilterFactory::New() );
    itk::ObjectFactoryBase::RegisterFactory( itk::GPUBSplineTransformFactory::New() );
    itk::ObjectFactoryBase::RegisterFactory( itk::GPUCastImageFilterFactory::New() );
  }

  typename GPUReaderType::Pointer GPUReader = GPUReaderType::New();
  GPUReader->SetFileName( _parameters.inputFileNames[0] );
  GPUReader->Update();

  // Create GPUTransform
  typename GPUTransformType::Pointer GPUTransform = GPUTransformType::New();
  GPUTransform->SetTransformDomainOrigin( inputOrigin );
  GPUTransform->SetTransformDomainDirection( inputDirection );
  GPUTransform->SetTransformDomainPhysicalDimensions( fixedDimensions );
  GPUTransform->SetTransformDomainMeshSize( meshSize );
  GPUTransform->SetParameters( parameters );

  // Test GPU
  typename GPUFilterType::Pointer GPUFilter = GPUFilterType::New();

  // Create GPU interpolator here
  GPUInterpolatorPointerType GPUInterpolator;
  if(_parameters.interpolator == "NearestNeighbor")
  {
    if(_parameters.GPUEnable)
    {
      itk::ObjectFactoryBase::RegisterFactory( itk::GPUNearestNeighborInterpolateImageFunctionFactory::New() );
    }
    typename GPUNearestNeighborInterpolatorType::Pointer GPUNearestNeighborInterpolator
      = GPUNearestNeighborInterpolatorType::New();
    GPUInterpolator = GPUNearestNeighborInterpolator;
  }
  else if(_parameters.interpolator == "Linear")
  {
    if(_parameters.GPUEnable)
    {
      itk::ObjectFactoryBase::RegisterFactory( itk::GPULinearInterpolateImageFunctionFactory::New() );
    }
    typename GPULinearInterpolatorType::Pointer GPULinearInterpolator
      = GPULinearInterpolatorType::New();
    GPUInterpolator = GPULinearInterpolator;

  }
  else if(_parameters.interpolator == "BSpline")
  {
    if(_parameters.GPUEnable)
    {
      itk::ObjectFactoryBase::RegisterFactory( itk::GPUBSplineInterpolateImageFunctionFactory::New() );
      itk::ObjectFactoryBase::RegisterFactory( itk::GPUBSplineDecompositionImageFilterFactory::New() );
    }
    typename GPUBSplineInterpolatorType::Pointer GPUBSplineInterpolator
      = GPUBSplineInterpolatorType::New();
    GPUInterpolator = GPUBSplineInterpolator;
  }

  bool updateException = false;
  itk::TimeProbe gputimer;
  gputimer.Start();

  if(!_parameters.skipGPU)
  {
    for(unsigned int i=0; i<_parameters.runTimes; i++)
    {
      GPUFilter->SetInput( GPUReader->GetOutput() );
      GPUFilter->SetTransform( GPUTransform );
      GPUFilter->SetInterpolator( GPUInterpolator );
      GPUFilter->SetDefaultPixelValue( defaultValue );
      GPUFilter->SetOutputSpacing( inputSpacing );
      GPUFilter->SetOutputOrigin( inputOrigin );
      GPUFilter->SetOutputDirection( inputDirection );
      GPUFilter->SetSize( inputSize );
      GPUFilter->SetOutputStartIndex( inputRegion.GetIndex() );

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

  // UnRegister GPUImage before using BinaryThresholdImageFilter,
  // Otherwise GPU memory will be allocated
  itk::ObjectFactoryBase::UnRegisterFactory( imageFactory );

  // Create masks from filter output based on default value,
  // We compute rms error using this masks, otherwise we get false response due to floating errors.
  typedef itk::Image<unsigned char, OutputImageType::ImageDimension> MaskImageType;
  typedef itk::BinaryThresholdImageFilter<OutputImageType, MaskImageType> ThresholdType;

  const typename OutputImageType::PixelType lower = minValue - 1; // avoid floating errors

  typename ThresholdType::Pointer thresholderCPU = ThresholdType::New();
  thresholderCPU->SetInput(CPUFilter->GetOutput());
  thresholderCPU->SetInsideValue(itk::NumericTraits<typename MaskImageType::PixelType>::One);
  thresholderCPU->SetLowerThreshold(lower);
  thresholderCPU->SetUpperThreshold(itk::NumericTraits<typename OutputImageType::PixelType>::max());
  thresholderCPU->Update();

  typename ThresholdType::Pointer thresholderGPU = ThresholdType::New();
  thresholderGPU->SetInput(GPUFilter->GetOutput());
  thresholderGPU->SetInsideValue(itk::NumericTraits<typename MaskImageType::PixelType>::One);
  thresholderGPU->SetLowerThreshold(lower);
  thresholderGPU->SetUpperThreshold(itk::NumericTraits<typename OutputImageType::PixelType>::max());
  thresholderGPU->Update();

  float diff = 0.0;
  unsigned int nPix = 0;
  if(!_parameters.skipCPU && !_parameters.skipGPU && !updateException)
  {
    // RMS Error check
    const double epsilon = 0.01;
    itk::ImageRegionIterator<OutputImageType> cit(CPUFilter->GetOutput(),
      CPUFilter->GetOutput()->GetLargestPossibleRegion());
    itk::ImageRegionIterator<OutputImageType> git(GPUFilter->GetOutput(),
      GPUFilter->GetOutput()->GetLargestPossibleRegion());
    itk::ImageRegionIterator<MaskImageType> mcit(thresholderCPU->GetOutput(),
      thresholderCPU->GetOutput()->GetLargestPossibleRegion());
    itk::ImageRegionIterator<MaskImageType> mgit(thresholderGPU->GetOutput(),
      thresholderGPU->GetOutput()->GetLargestPossibleRegion());
    for(cit.GoToBegin(), git.GoToBegin(), mcit.GoToBegin(), mgit.GoToBegin();
      !cit.IsAtEnd(); ++cit, ++git, ++mcit, ++mgit)
    {
      if((mcit.Get() == itk::NumericTraits<typename MaskImageType::PixelType>::One) &&
         (mgit.Get() == itk::NumericTraits<typename MaskImageType::PixelType>::One))
      {
        float err = vnl_math_abs((float)(cit.Get()) - (float)(git.Get()));
        //if(err > epsilon)
        //  std::cout << "CPU : " << (double)(cit.Get()) << ", GPU : " << (double)(git.Get()) << std::endl;
        diff += err*err;
        nPix++;
      }
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
    comments.append("Interpolator : ");
    comments.append(CPUInterpolator->GetNameOfClass());
    comments.append(", Transform : ");
    comments.append(CPUTransform->GetNameOfClass());

    if(updateException)
      comments.append(", Exception during update");

    itk::WriteLog<InputImageType>(
      _parameters.logFileName, ImageDim, inputSize, RMSError,
      testPassed, updateException,
      CPUFilter->GetNumberOfThreads(), _parameters.runTimes,
      CPUFilter->GetNameOfClass(),
      cputimer.GetMeanTime(), gputimer.GetMeanTime(),
      comments);
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
  std::cout << "  -in           input file names" << std::endl;
  std::cout << "  -inpar        input parameters file name" << std::endl;
  std::cout << "  [-out]        output file names.(outputCPU outputGPU)" << std::endl;
  std::cout << "  [-outlog]     output log file name, default 'CPUGPULog.txt'" << std::endl;
  std::cout << "  [-nooutput]   controls where output is created, default write output" << std::endl;
  std::cout << "  [-runtimes]   controls how many times filter will execute, default 1" << std::endl;
  std::cout << "  [-skipcpu]    skip running CPU part, default false" << std::endl;
  std::cout << "  [-skipgpu]    skip running GPU part, default false" << std::endl;
  std::cout << "  [-i]          interpolator, one of {NearestNeighbor, Linear, BSpline}, default NearestNeighbor\n";
  std::cout << "  [-rms]        rms error, default 0" << std::endl;
  std::cout << "  [-gpu]        use GPU, default 0" << std::endl;
  std::cout << "  [-threads]    number of threads, default maximum" << std::endl;
}
