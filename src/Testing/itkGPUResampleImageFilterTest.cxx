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
#include "itkGPUResampleImageFilter.h"

// GPU copiers
#include "itkGPUTransformCopier.h"
#include "itkGPUAdvancedCombinationTransformCopier.h"
#include "itkGPUInterpolatorCopier.h"

// GPU factory includes
#include "itkGPUImageFactory.h"
#include "itkGPUResampleImageFilterFactory.h"
#include "itkGPUCastImageFilterFactory.h"
#include "itkGPUAffineTransformFactory.h"
#include "itkGPUTranslationTransformFactory.h"
#include "itkGPUBSplineTransformFactory.h"
#include "itkGPUEuler3DTransformFactory.h"
#include "itkGPUSimilarity3DTransformFactory.h"
#include "itkGPUNearestNeighborInterpolateImageFunctionFactory.h"
#include "itkGPULinearInterpolateImageFunctionFactory.h"
#include "itkGPUBSplineInterpolateImageFunctionFactory.h"
#include "itkGPUBSplineDecompositionImageFilterFactory.h"

// elastix GPU factory includes
#include "itkGPUAdvancedCombinationTransformFactory.h"
#include "itkGPUAdvancedMatrixOffsetTransformBaseFactory.h"
#include "itkGPUAdvancedTranslationTransformFactory.h"
#include "itkGPUAdvancedBSplineDeformableTransformFactory.h"
#include "itkGPUAdvancedSimilarity3DTransformFactory.h"

// ITK include files
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionConstIterator.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"
#include "itkOutputWindow.h"
#include "itkTimeProbe.h"

// elastix include files
#include "itkCommandLineArgumentParser.h"

// Other include files
#include <iomanip> // setprecision, etc.
#include <sstream>

//------------------------------------------------------------------------------
// GetHelpString
std::string
GetHelpString( void )
{
  std::stringstream ss;

  ss << "Usage:" << std::endl
     << "itkGPUResampleImageFilterTest" << std::endl
     << "  -in           input file name\n"
     << "  -out          output file names.(outputCPU outputGPU)\n"
     << "  -rmse         acceptable rmse error\n"
     << "  [-i]          interpolator, one of {NearestNeighbor, Linear, BSpline}, default NearestNeighbor\n"
     << "  [-t]          transforms, one of {Affine, Translation, BSpline, Euler, Similarity}"
     << " or combinations with option \"-c\", default Affine\n"
     << "  [-c]          use combo transform, default false\n"
     << "  [-p]          parameter file for the B-spline transform\n"
     << "  [-threads]    number of threads, default maximum\n";
  return ss.str();
} // end GetHelpString()


//------------------------------------------------------------------------------
template< typename TransformType, typename CompositeTransformType >
void
PrintTransform( typename TransformType::Pointer & transform )
{
  std::cout << "Transform type: " << transform->GetNameOfClass();

  const CompositeTransformType * compositeTransform
    = dynamic_cast< const CompositeTransformType * >( transform.GetPointer() );

  if( compositeTransform )
  {
    std::cout << " [";
    for( std::size_t i = 0; i < compositeTransform->GetNumberOfTransforms(); i++ )
    {
      std::cout << compositeTransform->GetNthTransform( i )->GetNameOfClass();
      if( i != compositeTransform->GetNumberOfTransforms() - 1 )
      {
        std::cout << ", ";
      }
    }
    std::cout << "]" << std::endl;
  }
  else
  {
    std::cout << std::endl;
  }
}


//------------------------------------------------------------------------------
template< typename InputImageType >
typename InputImageType::PointType
ComputeCenterOfTheImage( const typename InputImageType::ConstPointer & image )
{
  const unsigned int Dimension = image->GetImageDimension();

  const typename InputImageType::SizeType size   = image->GetLargestPossibleRegion().GetSize();
  const typename InputImageType::IndexType index = image->GetLargestPossibleRegion().GetIndex();

  typedef itk::ContinuousIndex< double, InputImageType::ImageDimension > ContinuousIndexType;
  ContinuousIndexType centerAsContInd;
  for( std::size_t i = 0; i < Dimension; i++ )
  {
    centerAsContInd[ i ]
      = static_cast< double >( index[ i ] )
      + static_cast< double >( size[ i ] - 1 ) / 2.0;
  }

  typename InputImageType::PointType center;
  image->TransformContinuousIndexToPhysicalPoint( centerAsContInd, center );
  return center;
}


//------------------------------------------------------------------------------
template< typename InterpolatorType >
void
DefineInterpolator( typename InterpolatorType::Pointer & interpolator,
  const std::string & interpolatorName,
  const unsigned int splineOrderInterpolator )
{
  // Interpolator typedefs
  typedef typename InterpolatorType::InputImageType InputImageType;
  typedef typename InterpolatorType::CoordRepType   CoordRepType;
  typedef CoordRepType                              CoefficientType;

  // Typedefs for all interpolators
  typedef itk::NearestNeighborInterpolateImageFunction<
    InputImageType, CoordRepType > NearestNeighborInterpolatorType;
  typedef itk::LinearInterpolateImageFunction<
    InputImageType, CoordRepType > LinearInterpolatorType;
  typedef itk::BSplineInterpolateImageFunction<
    InputImageType, CoordRepType, CoefficientType > BSplineInterpolatorType;

  if( interpolatorName == "NearestNeighbor" )
  {
    typename NearestNeighborInterpolatorType::Pointer tmpInterpolator
                 = NearestNeighborInterpolatorType::New();
    interpolator = tmpInterpolator;
  }
  else if( interpolatorName == "Linear" )
  {
    typename LinearInterpolatorType::Pointer tmpInterpolator
                 = LinearInterpolatorType::New();
    interpolator = tmpInterpolator;
  }
  else if( interpolatorName == "BSpline" )
  {
    typename BSplineInterpolatorType::Pointer tmpInterpolator
      = BSplineInterpolatorType::New();
    tmpInterpolator->SetSplineOrder( splineOrderInterpolator );
    interpolator = tmpInterpolator;
  }
}


//------------------------------------------------------------------------------
template< typename AffineTransformType >
void
DefineAffineParameters( typename AffineTransformType::ParametersType & parameters )
{
  const unsigned int Dimension = AffineTransformType::InputSpaceDimension;

  // Setup parameters
  parameters.SetSize( Dimension * Dimension + Dimension );
  unsigned int par = 0;
  if( Dimension == 2 )
  {
    const double matrix[] =
    {
      0.9, 0.1, // matrix part
      0.2, 1.1, // matrix part
      0.0, 0.0, // translation
    };

    for( std::size_t i = 0; i < 6; i++ )
    {
      parameters[ par++ ] = matrix[ i ];
    }
  }
  else if( Dimension == 3 )
  {
    const double matrix[] =
    {
      1.0, -0.045, 0.02,   // matrix part
      0.0, 1.0, 0.0,       // matrix part
      -0.075, 0.09, 1.0,   // matrix part
      -3.02, 1.3, -0.045   // translation
    };

    for( std::size_t i = 0; i < 12; i++ )
    {
      parameters[ par++ ] = matrix[ i ];
    }
  }
}


//------------------------------------------------------------------------------
template< typename TranslationTransformType >
void
DefineTranslationParameters( const std::size_t transformIndex,
  typename TranslationTransformType::ParametersType & parameters )
{
  const std::size_t Dimension = TranslationTransformType::SpaceDimension;

  // Setup parameters
  parameters.SetSize( Dimension );
  for( std::size_t i = 0; i < Dimension; i++ )
  {
    parameters[ i ] = (double)i * (double)transformIndex + (double)transformIndex;
  }
}


//------------------------------------------------------------------------------
template< typename BSplineTransformType >
void
DefineBSplineParameters( const std::size_t transformIndex,
  typename BSplineTransformType::ParametersType & parameters,
  const typename BSplineTransformType::Pointer & transform,
  const std::string & parametersFileName )
{
  const unsigned int numberOfParameters = transform->GetNumberOfParameters();
  const unsigned int Dimension          = BSplineTransformType::SpaceDimension;
  const unsigned int numberOfNodes      = numberOfParameters / Dimension;

  parameters.SetSize( numberOfParameters );

  // Open file and read parameters
  std::ifstream infile;
  infile.open( parametersFileName.c_str() );

  // Skip number of elements to make unique coefficients per each transformIndex
  for( std::size_t n = 0; n < transformIndex; n++ )
  {
    double parValue;
    infile >> parValue;
  }

  // Read it
  for( std::size_t n = 0; n < numberOfNodes * Dimension; n++ )
  {
    double parValue;
    infile >> parValue;
    parameters[ n ] = parValue;
  }

  infile.close();
}


//------------------------------------------------------------------------------
template< typename EulerTransformType >
void
DefineEulerParameters( const std::size_t transformIndex,
  typename EulerTransformType::ParametersType & parameters )
{
  const std::size_t Dimension = EulerTransformType::InputSpaceDimension;

  // Setup parameters
  // 2D: angle 1, translation 2
  // 3D: 6 angle, translation 3
  parameters.SetSize( EulerTransformType::ParametersDimension );

  // Angle
  const double angle = (double)transformIndex * -0.05;

  std::size_t par = 0;
  if( Dimension == 2 )
  {
    // See implementation of Rigid2DTransform::SetParameters()
    parameters[ 0 ] = angle;
    ++par;
  }
  else if( Dimension == 3 )
  {
    // See implementation of Rigid3DTransform::SetParameters()
    for( std::size_t i = 0; i < 3; i++ )
    {
      parameters[ par ] = angle;
      ++par;
    }
  }

  for( std::size_t i = 0; i < Dimension; i++ )
  {
    parameters[ i + par ] = (double)i * (double)transformIndex + (double)transformIndex;
  }
}


//------------------------------------------------------------------------------
template< typename SimilarityTransformType >
void
DefineSimilarityParameters( const std::size_t transformIndex,
  typename SimilarityTransformType::ParametersType & parameters )
{
  const std::size_t Dimension = SimilarityTransformType::InputSpaceDimension;

  // Setup parameters
  // 2D: 2 translation, angle 1, scale 1
  // 3D: 3 translation, angle 3, scale 1
  parameters.SetSize( SimilarityTransformType::ParametersDimension );

  // Scale, Angle
  const double scale = ( (double)transformIndex + 1.0 ) * 0.05 + 1.0;
  const double angle = (double)transformIndex * -0.06;

  if( Dimension == 2 )
  {
    // See implementation of Similarity2DTransform::SetParameters()
    parameters[ 0 ] = scale;
    parameters[ 1 ] = angle;
  }
  else if( Dimension == 3 )
  {
    // See implementation of Similarity3DTransform::SetParameters()
    for( std::size_t i = 0; i < Dimension; i++ )
    {
      parameters[ i ] = angle;
    }
    parameters[ 6 ] = scale;
  }

  // Translation
  for( std::size_t i = 0; i < Dimension; i++ )
  {
    parameters[ i + Dimension ] = -1.0 * ( (double)i * (double)transformIndex + (double)transformIndex );
  }
}


//------------------------------------------------------------------------------
// This helper function completely set the transform
// We are using ITK elastix transforms:
// ITK transforms:
// TransformType, AffineTransformType, TranslationTransformType,
// BSplineTransformType, EulerTransformType, SimilarityTransformType
// elastix Transforms:
// AdvancedCombinationTransformType, AdvancedAffineTransformType,
// AdvancedTranslationTransformType, AdvancedBSplineTransformType,
// AdvancedEulerTransformType, AdvancedSimilarityTransformType
template< typename TransformType, typename AffineTransformType,
typename TranslationTransformType, typename BSplineTransformType,
typename EulerTransformType, typename SimilarityTransformType,
typename AdvancedCombinationTransformType, typename AdvancedAffineTransformType,
typename AdvancedTranslationTransformType, typename AdvancedBSplineTransformType,
typename AdvancedEulerTransformType, typename AdvancedSimilarityTransformType,
typename InputImageType >
void
SetTransform( const std::size_t transformIndex,
  const std::string & transformName,
  typename TransformType::Pointer & transform,
  typename AdvancedCombinationTransformType::Pointer & advancedTransform,
  const typename InputImageType::ConstPointer & image,
  std::vector< typename BSplineTransformType::ParametersType > & bsplineParameters,
  const std::string & parametersFileName )
{
  if( transformName == "Affine" )
  {
    if( advancedTransform.IsNull() )
    {
      // Create Affine transform
      typename AffineTransformType::Pointer affineTransform
        = AffineTransformType::New();

      // Define and set affine parameters
      typename AffineTransformType::ParametersType parameters;
      DefineAffineParameters< AffineTransformType >( parameters );
      affineTransform->SetParameters( parameters );

      transform = affineTransform;
    }
    else
    {
      // Create Advanced Affine transform
      typename AdvancedAffineTransformType::Pointer affineTransform
        = AdvancedAffineTransformType::New();
      advancedTransform->SetCurrentTransform( affineTransform );

      // Define and set advanced affine parameters
      typename AdvancedAffineTransformType::ParametersType parameters;
      DefineAffineParameters< AdvancedAffineTransformType >( parameters );
      affineTransform->SetParameters( parameters );
    }
  }
  else if( transformName == "Translation" )
  {
    if( advancedTransform.IsNull() )
    {
      // Create Translation transform
      typename TranslationTransformType::Pointer translationTransform
        = TranslationTransformType::New();

      // Define and set translation parameters
      typename TranslationTransformType::ParametersType parameters;
      DefineTranslationParameters< TranslationTransformType >
        ( transformIndex, parameters );
      translationTransform->SetParameters( parameters );

      transform = translationTransform;
    }
    else
    {
      // Create Advanced Translation transform
      typename AdvancedTranslationTransformType::Pointer translationTransform
        = AdvancedTranslationTransformType::New();
      advancedTransform->SetCurrentTransform( translationTransform );

      // Define and set advanced translation parameters
      typename AdvancedTranslationTransformType::ParametersType parameters;
      DefineTranslationParameters< AdvancedTranslationTransformType >
        ( transformIndex, parameters );
      translationTransform->SetParameters( parameters );
    }
  }
  else if( transformName == "BSpline" )
  {
    const unsigned int Dimension = image->GetImageDimension();
    const typename InputImageType::SpacingType inputSpacing     = image->GetSpacing();
    const typename InputImageType::PointType inputOrigin        = image->GetOrigin();
    const typename InputImageType::DirectionType inputDirection = image->GetDirection();
    const typename InputImageType::RegionType inputRegion       = image->GetBufferedRegion();
    const typename InputImageType::SizeType inputSize           = inputRegion.GetSize();

    typedef typename BSplineTransformType::MeshSizeType MeshSizeType;
    MeshSizeType gridSize;
    gridSize.Fill( 4 );

    typedef typename BSplineTransformType::PhysicalDimensionsType PhysicalDimensionsType;
    PhysicalDimensionsType gridSpacing;
    for( unsigned int d = 0; d < Dimension; d++ )
    {
      gridSpacing[ d ] = inputSpacing[ d ] * ( inputSize[ d ] - 1.0 );
    }

    if( advancedTransform.IsNull() )
    {
      // Create BSpline transform
      typename BSplineTransformType::Pointer bsplineTransform
        = BSplineTransformType::New();

      // Set grid properties
      bsplineTransform->SetTransformDomainOrigin( inputOrigin );
      bsplineTransform->SetTransformDomainDirection( inputDirection );
      bsplineTransform->SetTransformDomainPhysicalDimensions( gridSpacing );
      bsplineTransform->SetTransformDomainMeshSize( gridSize );

      // Define and set b-spline parameters
      typename BSplineTransformType::ParametersType parameters;
      DefineBSplineParameters< BSplineTransformType >
        ( transformIndex, parameters, bsplineTransform, parametersFileName );

      // Keep them in memory first by copying to the bsplineParameters
      bsplineParameters.push_back( parameters );
      const std::size_t indexAt = bsplineParameters.size() - 1;

      // Do not set parameters, the will be destroyed going out of scope
      // instead, set the ones from the bsplineParameters array
      bsplineTransform->SetParameters( bsplineParameters[ indexAt ] );

      transform = bsplineTransform;
    }
    else
    {
      // Create Advanced BSpline transform
      typename AdvancedBSplineTransformType::Pointer bsplineTransform
        = AdvancedBSplineTransformType::New();
      advancedTransform->SetCurrentTransform( bsplineTransform );

      // Set grid properties
      bsplineTransform->SetGridOrigin( inputOrigin );
      bsplineTransform->SetGridDirection( inputDirection );
      bsplineTransform->SetGridSpacing( gridSpacing );
      bsplineTransform->SetGridRegion( gridSize );

      // Define and set b-spline parameters
      typename AdvancedBSplineTransformType::ParametersType parameters;
      DefineBSplineParameters< AdvancedBSplineTransformType >
        ( transformIndex, parameters, bsplineTransform, parametersFileName );

      // Keep them in memory first by copying to the bsplineParameters
      bsplineParameters.push_back( parameters );
      const std::size_t indexAt = bsplineParameters.size() - 1;

      // Do not set parameters, the will be destroyed going out of scope
      // instead, set the ones from the bsplineParameters array
      bsplineTransform->SetParameters( bsplineParameters[ indexAt ] );
    }
  }
  else if( transformName == "Euler" )
  {
    // Compute center
    const typename InputImageType::PointType center
      = ComputeCenterOfTheImage< InputImageType >( image );

    if( advancedTransform.IsNull() )
    {
      // Create Euler transform
      typename EulerTransformType::Pointer eulerTransform
        = EulerTransformType::New();

      // Set center
      eulerTransform->SetCenter( center );

      // Define and set euler parameters
      typename EulerTransformType::ParametersType parameters;
      DefineEulerParameters< EulerTransformType >
        ( transformIndex, parameters );
      eulerTransform->SetParameters( parameters );

      transform = eulerTransform;
    }
    else
    {
      // Create Advanced Euler transform
      typename AdvancedEulerTransformType::Pointer eulerTransform
        = AdvancedEulerTransformType::New();
      advancedTransform->SetCurrentTransform( eulerTransform );

      // Set center
      eulerTransform->SetCenter( center );

      // Define and set advanced euler parameters
      typename AdvancedEulerTransformType::ParametersType parameters;
      DefineEulerParameters< AdvancedEulerTransformType >
        ( transformIndex, parameters );
      eulerTransform->SetParameters( parameters );
    }
  }
  else if( transformName == "Similarity" )
  {
    // Compute center
    const typename InputImageType::PointType center
      = ComputeCenterOfTheImage< InputImageType >( image );

    if( advancedTransform.IsNull() )
    {
      // Create Similarity transform
      typename SimilarityTransformType::Pointer similarityTransform
        = SimilarityTransformType::New();

      // Set center
      similarityTransform->SetCenter( center );

      // Define and set similarity parameters
      typename SimilarityTransformType::ParametersType parameters;
      DefineSimilarityParameters< SimilarityTransformType >
        ( transformIndex, parameters );
      similarityTransform->SetParameters( parameters );

      transform = similarityTransform;
    }
    else
    {
      // Create Advanced Similarity transform
      typename AdvancedSimilarityTransformType::Pointer similarityTransform
        = AdvancedSimilarityTransformType::New();
      advancedTransform->SetCurrentTransform( similarityTransform );

      // Set center
      similarityTransform->SetCenter( center );

      // Define and set advanced similarity parameters
      typename AdvancedSimilarityTransformType::ParametersType parameters;
      DefineSimilarityParameters< AdvancedSimilarityTransformType >
        ( transformIndex, parameters );
      similarityTransform->SetParameters( parameters );
    }
  }
}


//------------------------------------------------------------------------------
// This test compares the CPU with the GPU version of the ResampleImageFilter.
// The filter takes an input image and produces an output image.
// We compare the CPU and GPU output image using RMSE and speed.
//
// The following ITK interpolations are supported:
// itk::NearestNeighborInterpolateImageFunction
// itk::LinearInterpolateImageFunction
// itk::BSplineInterpolateImageFunction
//
// The following ITK transforms are supported:
// itk::CompositeTransform
// itk::AffineTransform
// itk::TranslationTransform
// itk::BSplineTransform
// itk::Euler3DTransform
// itk::Similarity3DTransform
//
// The following elastix transforms are supported:
// itk::AdvancedCombinationTransform
// itk::AdvancedMatrixOffsetTransformBase
// itk::AdvancedTranslationTransform
// itk::AdvancedBSplineDeformableTransform
// itk::AdvancedEuler3DTransform
// itk::AdvancedSimilarity3DTransform
//
int
main( int argc, char * argv[] )
{
  // Setup for debugging
  itk::SetupForDebugging();

  // Create and check OpenCL context
  if( !itk::CreateContext() )
  {
    return EXIT_FAILURE;
  }

  // Check for the device 'double' support
  itk::OpenCLContext::Pointer context = itk::OpenCLContext::GetInstance();
  if( !context->GetDefaultDevice().HasDouble() )
  {
    std::cerr << "Your OpenCL device: " << context->GetDefaultDevice().GetName()
              << ", does not support 'double' computations. Consider updating it." << std::endl;
    itk::ReleaseContext();
    return EXIT_FAILURE;
  }

  // Create a command line argument parser
  itk::CommandLineArgumentParser::Pointer parser = itk::CommandLineArgumentParser::New();
  parser->SetCommandLineArguments( argc, argv );
  parser->SetProgramHelpText( GetHelpString() );

  parser->MarkArgumentAsRequired( "-in", "The input filename" );
  parser->MarkArgumentAsRequired( "-out", "The output filenames" );
  parser->MarkArgumentAsRequired( "-rmse", "The acceptable rmse error" );

  itk::CommandLineArgumentParser::ReturnValue validateArguments = parser->CheckForRequiredArguments();

  if( validateArguments == itk::CommandLineArgumentParser::FAILED )
  {
    itk::ReleaseContext();
    return EXIT_FAILURE;
  }
  else if( validateArguments == itk::CommandLineArgumentParser::HELPREQUESTED )
  {
    itk::ReleaseContext();
    return EXIT_SUCCESS;
  }

  // Get command line arguments
  std::string inputFileName = "";
  parser->GetCommandLineArgument( "-in", inputFileName );

  std::vector< std::string > outputFileNames( 2, "" );
  parser->GetCommandLineArgument( "-out", outputFileNames );

  // Get acceptable rmse error
  double rmseError;
  parser->GetCommandLineArgument( "-rmse", rmseError );

  // interpolator argument
  std::string interpolator = "NearestNeighbor";
  parser->GetCommandLineArgument( "-i", interpolator );

  if( interpolator != "NearestNeighbor"
    && interpolator != "Linear"
    && interpolator != "BSpline" )
  {
    std::cerr << "ERROR: interpolator \"-i\" should be one of {NearestNeighbor, Linear, BSpline}." << std::endl;
    itk::ReleaseContext();
    return EXIT_FAILURE;
  }

  // transform argument
  const bool                 useComboTransform = parser->ArgumentExists( "-c" );
  std::vector< std::string > transforms;
  transforms.push_back( "Affine" );
  parser->GetCommandLineArgument( "-t", transforms );

  // check that use combo transform provided when used multiple transforms
  if( transforms.size() > 1 && !useComboTransform )
  {
    std::cerr << "ERROR: for multiple transforms option \"-c\" should provided." << std::endl;
    itk::ReleaseContext();
    return EXIT_FAILURE;
  }

  // check for supported transforms
  for( std::size_t i = 0; i < transforms.size(); i++ )
  {
    const std::string transformName = transforms[ i ];
    if( transformName != "Affine"
      && transformName != "Translation"
      && transformName != "BSpline"
      && transformName != "Euler"
      && transformName != "Similarity" )
    {
      std::cerr << "ERROR: transforms \"-t\" should be one of "
                << "{Affine, Translation, BSpline, Euler, Similarity}"
                << " or combination of them." << std::endl;
      itk::ReleaseContext();
      return EXIT_FAILURE;
    }
  }

  unsigned int runTimes           = 1;
  std::string  parametersFileName = "";
  for( std::size_t i = 0; i < transforms.size(); i++ )
  {
    if( transforms[ i ] == "BSpline" )
    {
      const bool retp = parser->GetCommandLineArgument( "-p", parametersFileName );
      if( !retp )
      {
        std::cerr << "ERROR: You should specify parameters file \"-p\" for the B-spline transform." << std::endl;
        itk::ReleaseContext();
        return EXIT_FAILURE;
      }
      // Faster B-spline tests
      runTimes = 1;
    }
  }

  // Threads.
  unsigned int maximumNumberOfThreads = itk::MultiThreader::GetGlobalDefaultNumberOfThreads();
  parser->GetCommandLineArgument( "-threads", maximumNumberOfThreads );
  itk::MultiThreader::SetGlobalMaximumNumberOfThreads( maximumNumberOfThreads );

  // Setup for debugging.
  itk::SetupForDebugging();

  const unsigned int splineOrderInterpolator = 3;
  std::cout << std::showpoint << std::setprecision( 4 );

  // Typedefs.
  const unsigned int Dimension = 3;
  typedef short                                    InputPixelType;
  typedef short                                    OutputPixelType;
  typedef itk::Image< InputPixelType, Dimension >  InputImageType;
  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;
  typedef InputImageType::SizeType::SizeValueType  SizeValueType;
  typedef typelist::MakeTypeList< short >::Type    OCLImageTypes;

  // CPU typedefs
  typedef float InterpolatorPrecisionType;
  typedef float ScalarType;
  typedef itk::ResampleImageFilter
    < InputImageType, OutputImageType, InterpolatorPrecisionType > FilterType;

  // Transform typedefs
  typedef itk::Transform< ScalarType, Dimension, Dimension > TransformType;
  typedef itk::AffineTransform< ScalarType, Dimension >      AffineTransformType;
  typedef itk::TranslationTransform< ScalarType, Dimension > TranslationTransformType;
  typedef itk::BSplineTransform< ScalarType, Dimension, 3 >  BSplineTransformType;
  typedef itk::Euler3DTransform< ScalarType >                EulerTransformType;
  typedef itk::Similarity3DTransform< ScalarType >           SimilarityTransformType;

  // Advanced transform typedefs
  typedef itk::AdvancedCombinationTransform< ScalarType, Dimension >
    AdvancedCombinationTransformType;
  typedef itk::AdvancedTransform< ScalarType, Dimension, Dimension >
    AdvancedTransformType;
  typedef itk::AdvancedMatrixOffsetTransformBase< ScalarType, Dimension, Dimension >
    AdvancedAffineTransformType;
  typedef itk::AdvancedTranslationTransform< ScalarType, Dimension >
    AdvancedTranslationTransformType;
  typedef itk::AdvancedBSplineDeformableTransform< ScalarType, Dimension, 3 >
    AdvancedBSplineTransformType;
  typedef itk::AdvancedEuler3DTransform< ScalarType >
    AdvancedEulerTransformType;
  typedef itk::AdvancedSimilarity3DTransform< ScalarType >
    AdvancedSimilarityTransformType;

  // Transform copiers
  typedef itk::GPUAdvancedCombinationTransformCopier< OCLImageTypes, OCLImageDims, AdvancedCombinationTransformType, ScalarType >
    AdvancedTransformCopierType;
  typedef itk::GPUTransformCopier< OCLImageTypes, OCLImageDims, TransformType, ScalarType > TransformCopierType;

  // Interpolate typedefs
  typedef itk::InterpolateImageFunction<
    InputImageType, InterpolatorPrecisionType >             InterpolatorType;
//  typedef itk::NearestNeighborInterpolateImageFunction<
//    InputImageType, InterpolatorPrecisionType >             NearestNeighborInterpolatorType;
//  typedef itk::LinearInterpolateImageFunction<
//    InputImageType, InterpolatorPrecisionType >             LinearInterpolatorType;
//  typedef itk::BSplineInterpolateImageFunction<
//    InputImageType, ScalarType, InterpolatorPrecisionType > BSplineInterpolatorType;

  // Interpolator copier
  typedef itk::GPUInterpolatorCopier< OCLImageTypes, OCLImageDims, InterpolatorType, InterpolatorPrecisionType >
    InterpolateCopierType;

  typedef itk::ImageFileReader< InputImageType >  ReaderType;
  typedef itk::ImageFileWriter< OutputImageType > WriterType;

  // CPU part
  ReaderType::Pointer       cpuReader;
  FilterType::Pointer       cpuFilter;
  InterpolatorType::Pointer cpuInterpolator;
  TransformType::Pointer    cpuTransform;

  // Keep BSpline transform parameters in memory
  typedef BSplineTransformType::ParametersType BSplineParametersType;
  std::vector< BSplineParametersType > bsplineParameters;

  // CPU Reader
  cpuReader = ReaderType::New();
  cpuReader->SetFileName( inputFileName );
  try
  {
    cpuReader->Update();
  }
  catch( itk::ExceptionObject & e )
  {
    std::cerr << "ERROR: Caught ITK exception during cpuReader->Update(): " << e << std::endl;
    itk::ReleaseContext();
    return EXIT_FAILURE;
  }

  // Construct and setup the resample filter
  cpuFilter = FilterType::New();

  typedef itk::Statistics::MersenneTwisterRandomVariateGenerator RandomNumberGeneratorType;
  RandomNumberGeneratorType::Pointer randomNum = RandomNumberGeneratorType::GetInstance();

  InputImageType::ConstPointer        inputImage     = cpuReader->GetOutput();
  const InputImageType::SpacingType   inputSpacing   = inputImage->GetSpacing();
  const InputImageType::PointType     inputOrigin    = inputImage->GetOrigin();
  const InputImageType::DirectionType inputDirection = inputImage->GetDirection();
  const InputImageType::RegionType    inputRegion    = inputImage->GetBufferedRegion();
  const InputImageType::SizeType      inputSize      = inputRegion.GetSize();

  OutputImageType::SpacingType   outputSpacing;
  OutputImageType::PointType     outputOrigin;
  OutputImageType::DirectionType outputDirection;
  OutputImageType::SizeType      outputSize;
  std::stringstream              s; s << std::setprecision( 4 ) << std::setiosflags( std::ios_base::fixed );
  double                         tmp1, tmp2;
  for( std::size_t i = 0; i < Dimension; i++ )
  {
    tmp1 = randomNum->GetUniformVariate( 0.9, 1.1 );
    tmp2 = inputSpacing[ i ] * tmp1;
    s << tmp2; s >> outputSpacing[ i ]; s.clear();

    tmp1 = randomNum->GetUniformVariate( -10.0, 10.0 );
    tmp2 = inputOrigin[ i ] + tmp1;
    s << tmp2; s >> outputOrigin[ i ]; s.clear();

    for( unsigned int j = 0; j < Dimension; j++ )
    {
      //tmp = randomNum->GetUniformVariate( 0.9 * inputOrigin[ i ], 1.1 *
      // inputOrigin[ i ] );
      outputDirection[ i ][ j ] = inputDirection[ i ][ j ];        // * tmp;
    }

    tmp1            = randomNum->GetUniformVariate( 0.9, 1.1 );
    outputSize[ i ] = itk::Math::Round< SizeValueType >( inputSize[ i ] * tmp1 );
  }

  cpuFilter->SetDefaultPixelValue( -1.0 );
  cpuFilter->SetOutputSpacing( outputSpacing );
  cpuFilter->SetOutputOrigin( outputOrigin );
  cpuFilter->SetOutputDirection( outputDirection );
  cpuFilter->SetSize( outputSize );
  cpuFilter->SetOutputStartIndex( inputRegion.GetIndex() );

  // Construct, select and setup transform
  if( !useComboTransform )
  {
    AdvancedCombinationTransformType::Pointer dummy;
    SetTransform<
    // ITK Transforms
    TransformType, AffineTransformType, TranslationTransformType,
    BSplineTransformType, EulerTransformType, SimilarityTransformType,
    // elastix Transforms
    AdvancedCombinationTransformType, AdvancedAffineTransformType, AdvancedTranslationTransformType,
    AdvancedBSplineTransformType, AdvancedEulerTransformType, AdvancedSimilarityTransformType,
    InputImageType >
      ( 0, transforms[ 0 ], cpuTransform, dummy, inputImage, bsplineParameters, parametersFileName );
  }
  else
  {
    AdvancedTransformType::Pointer            currentTransform;
    AdvancedCombinationTransformType::Pointer initialTransform;
    AdvancedCombinationTransformType::Pointer tmpTransform
                     = AdvancedCombinationTransformType::New();
    initialTransform = tmpTransform;
    cpuTransform     = tmpTransform;

    for( std::size_t i = 0; i < transforms.size(); i++ )
    {
      if( i == 0 )
      {
        SetTransform<
        // ITK Transforms
        TransformType, AffineTransformType, TranslationTransformType,
        BSplineTransformType, EulerTransformType, SimilarityTransformType,
        // elastix Transforms
        AdvancedCombinationTransformType, AdvancedAffineTransformType, AdvancedTranslationTransformType,
        AdvancedBSplineTransformType, AdvancedEulerTransformType, AdvancedSimilarityTransformType,
        InputImageType >
          ( i, transforms[ i ], cpuTransform, initialTransform, inputImage, bsplineParameters, parametersFileName );
      }
      else
      {
        AdvancedCombinationTransformType::Pointer initialNext
          = AdvancedCombinationTransformType::New();

        SetTransform<
        // ITK Transforms
        TransformType, AffineTransformType, TranslationTransformType,
        BSplineTransformType, EulerTransformType, SimilarityTransformType,
        // elastix Transforms
        AdvancedCombinationTransformType, AdvancedAffineTransformType, AdvancedTranslationTransformType,
        AdvancedBSplineTransformType, AdvancedEulerTransformType, AdvancedSimilarityTransformType,
        InputImageType >
          ( i, transforms[ i ], cpuTransform, initialNext, inputImage, bsplineParameters, parametersFileName );

        initialTransform->SetInitialTransform( initialNext );
        initialTransform = initialNext;
      }
    }
  }

  // Create CPU interpolator here
  DefineInterpolator< InterpolatorType >(
    cpuInterpolator, interpolator, splineOrderInterpolator );

  // Print info
  std::cout << "Testing the ResampleImageFilter, CPU vs GPU:\n";
  std::cout << "CPU/GPU transform interpolator #threads time speedup RMSE\n";

  // Time the filter, run on the CPU
  itk::TimeProbe cputimer;
  cputimer.Start();

  for( std::size_t i = 0; i < runTimes; i++ )
  {
    cpuFilter->SetInput( cpuReader->GetOutput() );
    cpuFilter->SetTransform( cpuTransform );
    cpuFilter->SetInterpolator( cpuInterpolator );
    try
    {
      cpuFilter->Update();
    }
    catch( itk::ExceptionObject & e )
    {
      std::cerr << "ERROR: Caught ITK exception during cpuFilter->Update(): " << e << std::endl;
      itk::ReleaseContext();
      return EXIT_FAILURE;
    }

    // Modify the filter, only not the last iteration
    if( i != runTimes - 1 )
    {
      cpuFilter->Modified();
    }
  }
  cputimer.Stop();

  std::cout << "CPU " << cpuTransform->GetNameOfClass()
            << " " << cpuInterpolator->GetNameOfClass()
            << " " << cpuFilter->GetNumberOfThreads()
            << " " << cputimer.GetMean() / runTimes << std::endl;

  /** Write the CPU result. */
  WriterType::Pointer cpuWriter = WriterType::New();
  cpuWriter->SetInput( cpuFilter->GetOutput() );
  cpuWriter->SetFileName( outputFileNames[ 0 ].c_str() );
  try
  {
    cpuWriter->Update();
  }
  catch( itk::ExceptionObject & e )
  {
    std::cerr << "ERROR: Caught ITK exception during cpuWriter->Update(): " << e << std::endl;
    itk::ReleaseContext();
    return EXIT_FAILURE;
  }

  // Register object factory for GPU image and filter
  // All these filters that are constructed after this point are
  // turned into a GPU filter.
  itk::GPUImageFactory2< OCLImageTypes, OCLImageDims >
  ::RegisterOneFactory();
  itk::GPUResampleImageFilterFactory2< OCLImageTypes, OCLImageTypes, OCLImageDims >
  ::RegisterOneFactory();
  itk::GPUCastImageFilterFactory2< OCLImageTypes, OCLImageTypes, OCLImageDims >
  ::RegisterOneFactory();

  // Transforms factory registration
  itk::GPUAffineTransformFactory2< OCLImageDims >::RegisterOneFactory();
  itk::GPUTranslationTransformFactory2< OCLImageDims >::RegisterOneFactory();
  itk::GPUBSplineTransformFactory2< OCLImageDims >::RegisterOneFactory();
  itk::GPUEuler3DTransformFactory2< OCLImageDims >::RegisterOneFactory();
  itk::GPUSimilarity3DTransformFactory2< OCLImageDims >::RegisterOneFactory();

  // Interpolators factory registration
  itk::GPUNearestNeighborInterpolateImageFunctionFactory2< OCLImageTypes, OCLImageDims >
  ::RegisterOneFactory();
  itk::GPULinearInterpolateImageFunctionFactory2< OCLImageTypes, OCLImageDims >
  ::RegisterOneFactory();
  itk::GPUBSplineInterpolateImageFunctionFactory2< OCLImageTypes, OCLImageDims >
  ::RegisterOneFactory();
  itk::GPUBSplineDecompositionImageFilterFactory2< OCLImageTypes, OCLImageTypes, OCLImageDims >
  ::RegisterOneFactory();

  // Advanced transforms factory registration
  itk::GPUAdvancedCombinationTransformFactory2< OCLImageDims >::RegisterOneFactory();
  itk::GPUAdvancedMatrixOffsetTransformBaseFactory2< OCLImageDims >::RegisterOneFactory();
  itk::GPUAdvancedTranslationTransformFactory2< OCLImageDims >::RegisterOneFactory();
  itk::GPUAdvancedBSplineDeformableTransformFactory2< OCLImageDims >::RegisterOneFactory();
  //itk::GPUAdvancedEuler3DTransformFactory2< OCLImageDims >::RegisterOneFactory();
  itk::GPUAdvancedSimilarity3DTransformFactory2< OCLImageDims >::RegisterOneFactory();

  // GPU part
  ReaderType::Pointer       gpuReader;
  FilterType::Pointer       gpuFilter;
  InterpolatorType::Pointer gpuInterpolator;
  TransformType::Pointer    gpuTransform;

  // Construct the filter
  // Use a try/catch, because construction of this filter will trigger
  // OpenCL compilation, which may fail.
  try
  {
    gpuFilter = FilterType::New();
    itk::ITKObjectEnableWarnings( gpuFilter.GetPointer() );
  }
  catch( itk::ExceptionObject & e )
  {
    std::cerr << "Caught ITK exception during gpuFilter::New(): " << e << std::endl;
    itk::ReleaseContext();
    return EXIT_FAILURE;
  }

  gpuFilter->SetDefaultPixelValue( -1.0 );
  gpuFilter->SetOutputSpacing( outputSpacing );
  gpuFilter->SetOutputOrigin( outputOrigin );
  gpuFilter->SetOutputDirection( outputDirection );
  gpuFilter->SetSize( outputSize );
  gpuFilter->SetOutputStartIndex( inputRegion.GetIndex() );

  // Also need to re-construct the image reader, so that it now
  // reads a GPUImage instead of a normal image.
  // Otherwise, you will get an exception when running the GPU filter:
  // "ERROR: The GPU InputImage is NULL. Filter unable to perform."
  gpuReader = ReaderType::New();
  gpuReader->SetFileName( inputFileName );
  try
  {
    gpuReader->Update(); // needed?
  }
  catch( itk::ExceptionObject & e )
  {
    std::cerr << "ERROR: Caught ITK exception during cpuReader->Update(): " << e << std::endl;
    itk::ReleaseContext();
    return EXIT_FAILURE;
  }

  try
  {
    if( !useComboTransform )
    {
      TransformCopierType::Pointer copier = TransformCopierType::New();
      copier->SetInputTransform( cpuTransform );
      copier->SetExplicitMode( false );
      copier->Update();
      gpuTransform = copier->GetModifiableOutput();
    }
    else
    {
      // Get CPU AdvancedCombinationTransform
      const AdvancedCombinationTransformType * CPUAdvancedCombinationTransform
        = dynamic_cast< const AdvancedCombinationTransformType * >( cpuTransform.GetPointer() );
      if( CPUAdvancedCombinationTransform )
      {
        AdvancedTransformCopierType::Pointer copier = AdvancedTransformCopierType::New();
        copier->SetInputTransform( CPUAdvancedCombinationTransform );
        copier->SetExplicitMode( false );
        copier->Update();
        gpuTransform = copier->GetModifiableOutput();
      }
      else
      {
        std::cerr << "ERROR: Unable to retrieve CPU AdvancedCombinationTransform." << std::endl;
        itk::ReleaseContext();
        return EXIT_FAILURE;
      }
    }
  }
  catch( itk::ExceptionObject & e )
  {
    std::cerr << "ERROR: Caught ITK exception during copy transforms: " << e << std::endl;
    itk::ReleaseContext();
    return EXIT_FAILURE;
  }

  // Create GPU copy for interpolator here
  InterpolateCopierType::Pointer interpolateCopier = InterpolateCopierType::New();
  interpolateCopier->SetInputInterpolator( cpuInterpolator );
  interpolateCopier->SetExplicitMode( false );
  interpolateCopier->Update();
  gpuInterpolator = interpolateCopier->GetModifiableOutput();

  // Time the filter, run on the GPU
  itk::TimeProbe gputimer;
  gputimer.Start();
  for( std::size_t i = 0; i < runTimes; i++ )
  {
    try
    {
      gpuFilter->SetInput( gpuReader->GetOutput() );
      gpuFilter->SetTransform( gpuTransform );
      gpuFilter->SetInterpolator( gpuInterpolator );
    }
    catch( itk::ExceptionObject & e )
    {
      std::cerr << "ERROR: " << e << std::endl;
      itk::ReleaseContext();
      return EXIT_FAILURE;
    }

    try
    {
      gpuFilter->Update();
    }
    catch( itk::ExceptionObject & e )
    {
      std::cerr << "ERROR: " << e << std::endl;
      itk::ReleaseContext();
      return EXIT_FAILURE;
    }
    // Modify the filter, only not the last iteration
    if( i != runTimes - 1 )
    {
      gpuFilter->Modified();
    }
  }
  gputimer.Stop();

  std::cout << "GPU " << cpuTransform->GetNameOfClass()
            << " " << cpuInterpolator->GetNameOfClass()
            << " x " << gputimer.GetMean() / runTimes
            << " " << cputimer.GetMean() / gputimer.GetMean();

  /** Write the GPU result. */
  WriterType::Pointer gpuWriter = WriterType::New();
  gpuWriter->SetInput( gpuFilter->GetOutput() );
  gpuWriter->SetFileName( outputFileNames[ 1 ].c_str() );
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
  if( RMSerror > rmseError )
  {
    std::cerr << "ERROR: the RMSE between the CPU and GPU results is "
              << RMSerror << ", which is larger than the expected "
              << rmseError << std::endl;
    itk::ReleaseContext();
    return EXIT_FAILURE;
  }

  // End program.
  itk::ReleaseContext();
  return EXIT_SUCCESS;
}
