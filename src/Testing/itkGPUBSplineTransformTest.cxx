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
#include "itkOpenCLUtil.h"

// GPU include files
#pragma warning(push)
// warning C4355: 'this' : used in base member initializer list
#pragma warning(disable:4355)
#include "itkGPUBSplineTransform.h"
#include "itkGPUCastImageFilter.h"
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
      outputIndex = 0;

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
    std::vector<std::string> inputFileNames;
    std::string inputParametersFileName;
    std::vector<std::string> outputFileNames;
    unsigned int outputIndex;
    std::string logFileName;

    // Filter
  };
}

//------------------------------------------------------------------------------
// GPU explicit synchronization helper function
template<class ImageToImageFilterType, class OutputImageType>
void GPUExplicitSync(typename ImageToImageFilterType::Pointer &filter,
                     const bool filterUpdate,
                     const bool lockCPU,
                     const bool lockGPU)
{
  if(filterUpdate)
  {
    filter->Update();
  }

  typedef typename OutputImageType::PixelType OutputImagePixelType;
  typedef itk::GPUImage<OutputImagePixelType, OutputImageType::ImageDimension> GPUOutputImageType;
  GPUOutputImageType *GPUOutput = dynamic_cast<GPUOutputImageType *>(filter->GetOutput());
  if(GPUOutput)
  {
    GPUOutput->UpdateBuffers();
    GPUOutput->GetGPUDataManager()->SetCPUBufferLock(lockCPU);
    GPUOutput->GetGPUDataManager()->SetGPUBufferLock(lockGPU);
  }
}

//------------------------------------------------------------------------------
template <class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void CopyCoefficientImagesToGPU(
                                itk::BSplineTransform<TScalarType, NDimensions, VSplineOrder> *transform,
                                itk::FixedArray<typename itk::GPUImage<TScalarType, NDimensions>::Pointer, NDimensions> &coefficientArray)
{
  // CPU Typedefs
  typedef itk::BSplineTransform<TScalarType, NDimensions, VSplineOrder> BSplineTransformType;
  typedef typename BSplineTransformType::ImageType                      TransformCoefficientImageType;
  typedef typename BSplineTransformType::ImagePointer                   TransformCoefficientImagePointer;
  typedef typename BSplineTransformType::CoefficientImageArray          CoefficientImageArray;

  // GPU Typedefs
  typedef itk::GPUImage<TScalarType, NDimensions>                       GPUTransformCoefficientImageType;
  typedef typename GPUTransformCoefficientImageType::Pointer            GPUTransformCoefficientImagePointer;

  const CoefficientImageArray coefficientImageArray = transform->GetCoefficientImages();

  // Typedef for caster
  typedef itk::CastImageFilter<TransformCoefficientImageType, GPUTransformCoefficientImageType> CasterType;

  for(unsigned int i=0; i<coefficientImageArray.Size(); i++)
  {
    TransformCoefficientImagePointer coefficients = coefficientImageArray[i];

    GPUTransformCoefficientImagePointer GPUCoefficients = GPUTransformCoefficientImageType::New();
    GPUCoefficients->CopyInformation(coefficients);
    GPUCoefficients->SetRegions(coefficients->GetBufferedRegion());
    GPUCoefficients->Allocate();

    // Create caster
    typename CasterType::Pointer caster = CasterType::New();
    caster->SetInput( coefficients );
    caster->GraftOutput( GPUCoefficients );
    caster->Update();

    GPUExplicitSync<CasterType, GPUTransformCoefficientImageType>( caster, false, true, true );

    coefficientArray[i] = GPUCoefficients;
  }
}


//------------------------------------------------------------------------------
template<class InputImageType, class OutputImageType>
int ProcessImage(const Parameters &_parameters);

// Testing GPU BSplineTransform initialization

// 1D:
// -in D:\\work\\elastix-ext\\ITK4OpenCL\\data\\image-512-1D.mha -inpar D:\\work\\elastix-ext\\ITK4OpenCL\\data\\BSplineDisplacements2-3D.txt -out D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test7\\image-512-1D-out-cpu.mha D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test7\\image-512-1D-out-gpu.mha -gpu -rsm 0.0
// 2D:
// -in D:\\work\\elastix-ext\\ITK4OpenCL\\data\\image-256x256-2D.mha -inpar D:\\work\\elastix-ext\\ITK4OpenCL\\data\\BSplineDisplacements5-2D.txt -out D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test7\\image-256x256-2D-out-cpu.mha D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test7\\image-256x256-2D-out-gpu.mha -gpu -rsm 0.0
// 3D:
// -in D:\\work\\elastix-ext\\ITK4OpenCL\\data\\image-64x64x64-3D.mha -inpar D:\\work\\elastix-ext\\ITK4OpenCL\\data\\BSplineDisplacements2-3D.txt -out D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test7\\image-64x64x64-3D-out-cpu.mha D:\\work\\elastix-ext\\ITK4OpenCL\\data\\test7\\image-64x64x64-3D-out-gpu.mha -gpu -rsm 0.0

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
  parser->GetCommandLineArgument("-outindex", parameters.outputIndex);
  parser->GetCommandLineArgument("-outlog", parameters.logFileName);
  const bool retruntimes = parser->GetCommandLineArgument("-runtimes", parameters.runTimes);

  parser->GetCommandLineArgument("-rsm", parameters.RMSError);
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
    run( ProcessImage, short, float, 1 );
    //run( ProcessImage, float, float, 1 );

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
  const unsigned int SpaceDimension = ImageDim;

  typedef typename InputImageType::PixelType  InputPixelType;
  typedef typename OutputImageType::PixelType OutputPixelType;

  // CPU Typedefs
  typedef itk::BSplineTransform<float, SpaceDimension, 3>  CPUTransformType;
  typedef typename CPUTransformType::CoefficientImageArray CPUCoefficientImageArray;
  typedef typename CPUTransformType::ImageType             CPUCoefficientImage;
  typedef typename CPUCoefficientImage::Pointer            CPUCoefficientImagePointer;

  // CPU reader
  typedef itk::ImageFileReader<InputImageType> CPUReaderType;

  // GPU Typedefs
  typedef itk::BSplineTransform<float, SpaceDimension, 3>  GPUTransformType;
  typedef typename GPUTransformType::CoefficientImageArray GPUCoefficientImageArray;
  typedef typename GPUTransformType::ImageType             GPUCoefficientImage;
  typedef typename GPUCoefficientImage::Pointer            GPUCoefficientImagePointer;

  typedef typename GPUTransformType::ImageType::Pointer CPUCoefficientImagePointer;

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

  itk::TimeProbe cputimer;
  cputimer.Start();

  if(!_parameters.skipCPU)
  {
    CPUTransform->SetParameters( parameters );
  }

  cputimer.Stop();
  std::cout << "CPU " << CPUTransform->GetNameOfClass() << " took " << cputimer.GetMeanTime() << " seconds." << std::endl;

  if(_parameters.GPUEnable)
  {
    // register object factory for GPU image and filter
    itk::ObjectFactoryBase::RegisterFactory( itk::GPUImageFactory::New() );
    itk::ObjectFactoryBase::RegisterFactory( itk::GPUBSplineTransformFactory::New() );
    itk::ObjectFactoryBase::RegisterFactory( itk::GPUCastImageFilterFactory::New() );
  }

  // Create GPUTransform
  typename GPUTransformType::Pointer GPUTransform = GPUTransformType::New();
  GPUTransform->SetTransformDomainOrigin( inputOrigin );
  GPUTransform->SetTransformDomainDirection( inputDirection );
  GPUTransform->SetTransformDomainPhysicalDimensions( fixedDimensions );
  GPUTransform->SetTransformDomainMeshSize( meshSize );

  itk::TimeProbe gputimer;
  gputimer.Start();
  bool updateException = false;

  if(!_parameters.skipGPU)
  {
    GPUTransform->SetParameters( parameters );
  }

  // Define BSplineTransformCoefficientImageArray
  typedef itk::GPUImage<float, InputImageType::ImageDimension> GPUBSplineTransformCoefficientImageType;
  typedef typename GPUBSplineTransformCoefficientImageType::Pointer GPUBSplineTransformCoefficientImagePointer;
  typedef itk::FixedArray<GPUBSplineTransformCoefficientImagePointer, InputImageType::ImageDimension> BSplineTransformCoefficientImageArray;

  BSplineTransformCoefficientImageArray coefficientArray;

  if(!_parameters.skipGPU)
  {
    try
    {
      CopyCoefficientImagesToGPU<float, InputImageType::ImageDimension, 3>(
        GPUTransform.GetPointer(), coefficientArray);
    }
    catch( itk::ExceptionObject &e )
    {
      std::cerr << "Caught ITK exception during GPUFilter->Update(): " << e << std::endl;
      updateException = updateException || true;
    }
  }

  const CPUCoefficientImageArray cpuCoefficientImageArray = CPUTransform->GetCoefficientImages();

  gputimer.Stop();

  if(!_parameters.skipGPU)
  {  
    std::cout << "GPU " << GPUTransform->GetNameOfClass() << " took " << gputimer.GetMeanTime() << " seconds." << std::endl;
  }

  bool testPassed = false;
  std::vector<float> RMSErrors;
  for(unsigned int i=0; i<cpuCoefficientImageArray.Size(); i++)
  {
    CPUCoefficientImagePointer cpucoefficients = cpuCoefficientImageArray[i];
    GPUBSplineTransformCoefficientImagePointer gpucoefficients = coefficientArray[i];

    // RMS Error check
    const double epsilon = 0.01;
    float diff = 0.0;
    unsigned int nPix = 0;
    if(!_parameters.skipCPU && !_parameters.skipGPU && !updateException)
    {
      itk::ImageRegionConstIterator<CPUCoefficientImage> cit(cpucoefficients, cpucoefficients->GetLargestPossibleRegion());
      itk::ImageRegionIterator<GPUBSplineTransformCoefficientImageType> git(gpucoefficients, gpucoefficients->GetLargestPossibleRegion());
      for(cit.GoToBegin(), git.GoToBegin(); !cit.IsAtEnd(); ++cit, ++git)
      {
        float err = vnl_math_abs((float)(cit.Get()) - (float)(git.Get()));
        diff += err*err;
        nPix++;
      }
    }

    float RMSError = 0.0;
    if(!_parameters.skipCPU && !_parameters.skipGPU && !updateException)
    {
      RMSError = sqrt( diff / (float)nPix );
    }

    RMSErrors.push_back(RMSError);
    if (!updateException)
      testPassed |= (RMSError <= _parameters.RMSError);
  }

  // Write output
  if(_parameters.outputWrite)
  {
    if (_parameters.outputIndex > cpuCoefficientImageArray.Size())
    {
      std::cerr << "ERROR: The outputIndex " << _parameters.outputIndex
        << " larger than coefficient array size." << std::endl;
      return EXIT_FAILURE;
    }

    CPUCoefficientImagePointer cpucoefficients = cpuCoefficientImageArray[_parameters.outputIndex];
    GPUBSplineTransformCoefficientImagePointer gpucoefficients = coefficientArray[_parameters.outputIndex];

    if(!_parameters.skipCPU)
    {
      // Write output CPU image
      typedef itk::ImageFileWriter<CPUCoefficientImage> CPUWriterType;
      typename CPUWriterType::Pointer writerCPU = CPUWriterType::New();
      writerCPU->SetInput( cpucoefficients );
      writerCPU->SetFileName( _parameters.outputFileNames[0] );
      writerCPU->Update();
    }

    if(!_parameters.skipGPU && !updateException)
    {
      // Write output GPU image
      typedef itk::ImageFileWriter<GPUBSplineTransformCoefficientImageType> GPUWriterType;
      typename GPUWriterType::Pointer writerGPU = GPUWriterType::New();
      writerGPU->SetInput( gpucoefficients );
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
      _parameters.logFileName, ImageDim, inputSize,
      RMSErrors[_parameters.outputIndex],
      testPassed, updateException,
      1, _parameters.runTimes,
      GPUTransform->GetNameOfClass(),
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
  std::cout << "  -in           input file names" << std::endl;
  std::cout << "  -inpar        input parameters file name" << std::endl;
  std::cout << "  [-out]        output file names.(outputCPU outputGPU)" << std::endl;
  std::cout << "  [-outindex]   output index" << std::endl;
  std::cout << "  [-outlog]     output log file name, default 'CPUGPULog.txt'" << std::endl;
  std::cout << "  [-nooutput]   controls where output is created, default write output" << std::endl;
  std::cout << "  [-runtimes]   controls how many times filter will execute, default 1" << std::endl;
  std::cout << "  [-skipcpu]    skip running CPU part, default false" << std::endl;
  std::cout << "  [-skipgpu]    skip running GPU part, default false" << std::endl;
  std::cout << "  [-rms]        rms error, default 0" << std::endl;
  std::cout << "  [-gpu]        use GPU, default 0" << std::endl;
  std::cout << "  [-threads]    number of threads, default maximum" << std::endl;
}
