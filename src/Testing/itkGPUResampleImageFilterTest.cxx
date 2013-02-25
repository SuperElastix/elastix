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
#include "itkGPUResampleImageFilter.h"
#include "itkGPUExplicitSynchronization.h"
#include "itkOpenCLUtil.h" // IsGPUAvailable()
#include "elxTestHelper.h"

// ITK GPU transforms
#include "itkGPUAffineTransform.h"
#include "itkGPUTranslationTransform.h"
#include "itkGPUBSplineTransform.h"
#include "itkGPUEuler3DTransform.h"
#include "itkGPUSimilarity3DTransform.h"

// elastix GPU transforms
#include "itkGPUAdvancedCombinationTransform.h"
#include "itkGPUAdvancedMatrixOffsetTransformBase.h"
#include "itkGPUAdvancedTranslationTransform.h"
#include "itkGPUAdvancedBSplineDeformableTransform.h"
#include "itkGPUAdvancedEuler3DTransform.h"
#include "itkGPUAdvancedSimilarity3DTransform.h"

// ITK GPU interpolate functions
#include "itkGPUNearestNeighborInterpolateImageFunction.h"
#include "itkGPULinearInterpolateImageFunction.h"
#include "itkGPUBSplineInterpolateImageFunction.h"
#include "itkGPUBSplineDecompositionImageFilter.h"

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
std::string GetHelpString( void )
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
template< class TransformType, class CompositeTransformType >
void PrintTransform( typename TransformType::Pointer & transform )
{
  std::cout << "Transform type: " << transform->GetNameOfClass();

  const CompositeTransformType *compositeTransform =
    dynamic_cast< const CompositeTransformType * >( transform.GetPointer() );

  if ( compositeTransform )
  {
    std::cout << " [";
    for ( std::size_t i = 0; i < compositeTransform->GetNumberOfTransforms(); i++ )
    {
      std::cout << compositeTransform->GetNthTransform( i )->GetNameOfClass();
      if ( i != compositeTransform->GetNumberOfTransforms() - 1 )
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
template< class InputImageType >
typename InputImageType::PointType
ComputeCenterOfTheImage( const typename InputImageType::ConstPointer & image )
{
  const unsigned int Dimension = image->GetImageDimension();

  const typename InputImageType::SizeType size   = image->GetLargestPossibleRegion().GetSize();
  const typename InputImageType::IndexType index = image->GetLargestPossibleRegion().GetIndex();

  typedef itk::ContinuousIndex< double, InputImageType::ImageDimension > ContinuousIndexType;
  ContinuousIndexType centerAsContInd;
  for ( std::size_t i = 0; i < Dimension; i++ )
  {
    centerAsContInd[i] =
      static_cast< double >( index[i] )
      + static_cast< double >( size[i] - 1 ) / 2.0;
  }

  typename InputImageType::PointType center;
  image->TransformContinuousIndexToPhysicalPoint( centerAsContInd, center );
  return center;
}

//------------------------------------------------------------------------------
template< class InterpolatorType, class ScalarType >
void DefineInterpolator( typename InterpolatorType::Pointer & interpolator,
                         const std::string interpolatorName,
                         const unsigned int splineOrderInterpolator )
{
  typedef typename InterpolatorType::InputImageType InputImageType;
  typedef typename InterpolatorType::CoordRepType   CoordRepType;
  typedef itk::NearestNeighborInterpolateImageFunction<
      InputImageType, CoordRepType > NearestNeighborInterpolatorType;
  typedef itk::LinearInterpolateImageFunction<
      InputImageType, CoordRepType > LinearInterpolatorType;
  typedef itk::BSplineInterpolateImageFunction<
      InputImageType, ScalarType, CoordRepType > BSplineInterpolatorType;

  if ( interpolatorName == "NearestNeighbor" )
  {
    typename NearestNeighborInterpolatorType::Pointer tmpInterpolator =
      NearestNeighborInterpolatorType::New();
    interpolator = tmpInterpolator;
  }
  else if ( interpolatorName == "Linear" )
  {
    typename LinearInterpolatorType::Pointer tmpInterpolator =
      LinearInterpolatorType::New();
    interpolator = tmpInterpolator;
  }
  else if ( interpolatorName == "BSpline" )
  {
    typename BSplineInterpolatorType::Pointer tmpInterpolator =
      BSplineInterpolatorType::New();
    tmpInterpolator->SetSplineOrder( splineOrderInterpolator );
    interpolator = tmpInterpolator;
  }
}

//------------------------------------------------------------------------------
template< class AffineTransformType >
void DefineAffineParameters( typename AffineTransformType::ParametersType & parameters )
{
  const unsigned int Dimension = AffineTransformType::InputSpaceDimension;

  // Setup parameters
  parameters.SetSize( Dimension * Dimension + Dimension );
  unsigned int par = 0;
  if ( Dimension == 2 )
  {
    const double matrix[] =
    {
      0.9, 0.1, // matrix part
      0.2, 1.1, // matrix part
      0.0, 0.0, // translation
    };

    for ( std::size_t i = 0; i < 6; i++ )
    {
      parameters[par++] = matrix[i];
    }
  }
  else if ( Dimension == 3 )
  {
    const double matrix[] =
    {
      1.0, -0.045, 0.02,   // matrix part
      0.0, 1.0, 0.0,       // matrix part
      -0.075, 0.09, 1.0,   // matrix part
      -3.02, 1.3, -0.045   // translation
    };

    for ( std::size_t i = 0; i < 12; i++ )
    {
      parameters[par++] = matrix[i];
    }
  }
}

//------------------------------------------------------------------------------
template< class TranslationTransformType >
void DefineTranslationParameters( const std::size_t transformIndex,
                                  typename TranslationTransformType::ParametersType & parameters )
{
  const std::size_t Dimension = TranslationTransformType::SpaceDimension;

  // Setup parameters
  parameters.SetSize( Dimension );
  for ( std::size_t i = 0; i < Dimension; i++ )
  {
    parameters[i] = (double)i * (double)transformIndex + (double)transformIndex;
  }
}

//------------------------------------------------------------------------------
template< class BSplineTransformType >
void DefineBSplineParameters( const std::size_t transformIndex,
                              typename BSplineTransformType::ParametersType & parameters,
                              const typename BSplineTransformType::Pointer & transform,
                              const std::string & parametersFileName )
{
  const unsigned int numberOfParameters = transform->GetNumberOfParameters();
  const unsigned int Dimension = BSplineTransformType::SpaceDimension;
  const unsigned int numberOfNodes = numberOfParameters / Dimension;

  parameters.SetSize( numberOfParameters );

  // Open file and read parameters
  std::ifstream infile;
  infile.open( parametersFileName.c_str() );

  // Skip number of elements to make unique coefficients per each transformIndex
  for ( std::size_t n = 0; n < transformIndex; n++ )
  {
    double parValue;
    infile >> parValue;
  }

  // Read it
  for ( std::size_t n = 0; n < numberOfNodes * Dimension; n++ )
  {
    double parValue;
    infile >> parValue;
    parameters[n] = parValue;
  }

  infile.close();
}

//------------------------------------------------------------------------------
template< class EulerTransformType >
void DefineEulerParameters( const std::size_t transformIndex,
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
  if ( Dimension == 2 )
  {
    // See implementation of Rigid2DTransform::SetParameters()
    parameters[0] = angle;
    ++par;
  }
  else if ( Dimension == 3 )
  {
    // See implementation of Rigid3DTransform::SetParameters()
    for ( std::size_t i = 0; i < 3; i++ )
    {
      parameters[par] = angle;
      ++par;
    }
  }

  for ( std::size_t i = 0; i < Dimension; i++ )
  {
    parameters[i + par] = (double)i * (double)transformIndex + (double)transformIndex;
  }
}

//------------------------------------------------------------------------------
template< class SimilarityTransformType >
void DefineSimilarityParameters( const std::size_t transformIndex,
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

  if ( Dimension == 2 )
  {
    // See implementation of Similarity2DTransform::SetParameters()
    parameters[0] = scale;
    parameters[1] = angle;
  }
  else if ( Dimension == 3 )
  {
    // See implementation of Similarity3DTransform::SetParameters()
    for ( std::size_t i = 0; i < Dimension; i++ )
    {
      parameters[i] = angle;
    }
    parameters[6] = scale;
  }

  // Translation
  for ( std::size_t i = 0; i < Dimension; i++ )
  {
    parameters[i + Dimension] = -1.0 * ( (double)i * (double)transformIndex + (double)transformIndex );
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
template< class TransformType, class AffineTransformType,
          class TranslationTransformType, class BSplineTransformType,
          class EulerTransformType, class SimilarityTransformType,
          class AdvancedCombinationTransformType, class AdvancedAffineTransformType,
          class AdvancedTranslationTransformType, class AdvancedBSplineTransformType,
          class AdvancedEulerTransformType, class AdvancedSimilarityTransformType,
          class InputImageType >
void SetTransform( const std::size_t transformIndex,
                   const std::string & transformName,
                   typename TransformType::Pointer & transform,
                   typename AdvancedCombinationTransformType::Pointer & advancedTransform,
                   const typename InputImageType::ConstPointer & image,
                   std::vector< typename BSplineTransformType::ParametersType > & bsplineParameters,
                   const std::string & parametersFileName )
{
  if ( transformName == "Affine" )
  {
    if ( advancedTransform.IsNull() )
    {
      // Create Affine transform
      typename AffineTransformType::Pointer affineTransform =
        AffineTransformType::New();

      // Define and set affine parameters
      typename AffineTransformType::ParametersType parameters;
      DefineAffineParameters< AffineTransformType >( parameters );
      affineTransform->SetParameters( parameters );

      transform = affineTransform;
    }
    else
    {
      // Create Advanced Affine transform
      typename AdvancedAffineTransformType::Pointer affineTransform =
        AdvancedAffineTransformType::New();
      advancedTransform->SetCurrentTransform( affineTransform );

      // Define and set advanced affine parameters
      typename AdvancedAffineTransformType::ParametersType parameters;
      DefineAffineParameters< AdvancedAffineTransformType >( parameters );
      affineTransform->SetParameters( parameters );
    }
  }
  else if ( transformName == "Translation" )
  {
    if ( advancedTransform.IsNull() )
    {
      // Create Translation transform
      typename TranslationTransformType::Pointer translationTransform =
        TranslationTransformType::New();

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
      typename AdvancedTranslationTransformType::Pointer translationTransform =
        AdvancedTranslationTransformType::New();
      advancedTransform->SetCurrentTransform( translationTransform );

      // Define and set advanced translation parameters
      typename AdvancedTranslationTransformType::ParametersType parameters;
      DefineTranslationParameters< AdvancedTranslationTransformType >
        ( transformIndex, parameters );
      translationTransform->SetParameters( parameters );
    }
  }
  else if ( transformName == "BSpline" )
  {
    const unsigned int Dimension = image->GetImageDimension();
    const typename InputImageType::SpacingType inputSpacing   = image->GetSpacing();
    const typename InputImageType::PointType inputOrigin    = image->GetOrigin();
    const typename InputImageType::DirectionType inputDirection = image->GetDirection();
    const typename InputImageType::RegionType inputRegion    = image->GetBufferedRegion();
    const typename InputImageType::SizeType inputSize      = inputRegion.GetSize();

    typedef typename BSplineTransformType::MeshSizeType MeshSizeType;
    MeshSizeType gridSize;
    gridSize.Fill( 4 );

    typedef typename BSplineTransformType::PhysicalDimensionsType PhysicalDimensionsType;
    PhysicalDimensionsType gridSpacing;
    for ( unsigned int d = 0; d < Dimension; d++ )
    {
      gridSpacing[d] = inputSpacing[d] * ( inputSize[d] - 1.0 );
    }

    if ( advancedTransform.IsNull() )
    {
      // Create BSpline transform
      typename BSplineTransformType::Pointer bsplineTransform =
        BSplineTransformType::New();

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
      bsplineTransform->SetParameters( bsplineParameters[indexAt] );

      transform = bsplineTransform;
    }
    else
    {
      // Create Advanced BSpline transform
      typename AdvancedBSplineTransformType::Pointer bsplineTransform =
        AdvancedBSplineTransformType::New();
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
      bsplineTransform->SetParameters( bsplineParameters[indexAt] );
    }
  }
  else if ( transformName == "Euler" )
  {
    // Compute center
    const typename InputImageType::PointType center =
      ComputeCenterOfTheImage< InputImageType >( image );

    if ( advancedTransform.IsNull() )
    {
      // Create Euler transform
      typename EulerTransformType::Pointer eulerTransform =
        EulerTransformType::New();

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
      typename AdvancedEulerTransformType::Pointer eulerTransform =
        AdvancedEulerTransformType::New();
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
  else if ( transformName == "Similarity" )
  {
    // Compute center
    const typename InputImageType::PointType center =
      ComputeCenterOfTheImage< InputImageType >( image );

    if ( advancedTransform.IsNull() )
    {
      // Create Similarity transform
      typename SimilarityTransformType::Pointer similarityTransform =
        SimilarityTransformType::New();

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
      typename AdvancedSimilarityTransformType::Pointer similarityTransform =
        AdvancedSimilarityTransformType::New();
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
// This helper function completely copy the transform
// We are using ITK transforms:
// AffineTransformType, TranslationTransformType, BSplineTransformType,
// EulerTransformType, SimilarityTransformType, CompositeTransform
template< class TransformType,
          class AffineTransformType, class TranslationTransformType, class BSplineTransformType,
          class EulerTransformType, class SimilarityTransformType >
void CopyTransform( const typename TransformType::Pointer & transformFrom,
                    typename TransformType::Pointer & transformTo,
                    const std::vector< typename BSplineTransformType::ParametersType > & bsplineParameters )
{
  // Try Affine
  typename AffineTransformType::Pointer affine =
    dynamic_cast< AffineTransformType * >( transformFrom.GetPointer() );

  if ( affine )
  {
    // Create Affine transform
    typename AffineTransformType::Pointer affineTransform =
      AffineTransformType::New();

    affineTransform->SetFixedParameters( affine->GetFixedParameters() );
    affineTransform->SetParameters( affine->GetParameters() );
    transformTo = affineTransform;

    return;
  }

  // Try Translation
  typename TranslationTransformType::Pointer translation =
    dynamic_cast< TranslationTransformType * >( transformFrom.GetPointer() );

  if ( translation )
  {
    // Create Translation transform
    typename TranslationTransformType::Pointer translationTransform =
      TranslationTransformType::New();

    translationTransform->SetFixedParameters( translation->GetFixedParameters() );
    translationTransform->SetParameters( translation->GetParameters() );
    transformTo = translationTransform;

    return;
  }

  // Try BSpline
  typename BSplineTransformType::Pointer bspline =
    dynamic_cast< BSplineTransformType * >( transformFrom.GetPointer() );

  if ( bspline )
  {
    // Create BSpline transform
    typename BSplineTransformType::Pointer bsplineTransform =
      BSplineTransformType::New();

    // Set grid properties
    bsplineTransform->SetTransformDomainOrigin(
      bspline->GetTransformDomainOrigin() );
    bsplineTransform->SetTransformDomainDirection(
      bspline->GetTransformDomainDirection() );
    bsplineTransform->SetTransformDomainPhysicalDimensions(
      bspline->GetTransformDomainPhysicalDimensions() );
    bsplineTransform->SetTransformDomainMeshSize(
      bspline->GetTransformDomainMeshSize() );

    // Alternative way
    //bsplineTransform->SetFixedParameters( bspline->GetFixedParameters() );
    //bsplineTransform->SetCoefficientImages( bspline->GetCoefficientImages() );

    // Define and set b-spline parameters
    const typename BSplineTransformType::ParametersType parameters =
      bsplineParameters[0];
    bsplineTransform->SetParameters( parameters );
    transformTo = bsplineTransform;

    return;
  }

  // Try Euler
  typename EulerTransformType::Pointer euler =
    dynamic_cast< EulerTransformType * >( transformFrom.GetPointer() );

  if ( euler )
  {
    // Create Euler transform
    typename EulerTransformType::Pointer eulerTransform =
      EulerTransformType::New();

    eulerTransform->SetFixedParameters( euler->GetFixedParameters() );
    eulerTransform->SetParameters( euler->GetParameters() );
    transformTo = eulerTransform;

    return;
  }

  // Try Similarity
  typename SimilarityTransformType::Pointer similarity =
    dynamic_cast< SimilarityTransformType * >( transformFrom.GetPointer() );

  if ( similarity )
  {
    // Create Similarity transform
    typename SimilarityTransformType::Pointer similarityTransform =
      SimilarityTransformType::New();

    similarityTransform->SetFixedParameters( similarity->GetFixedParameters() );
    similarityTransform->SetParameters( similarity->GetParameters() );
    transformTo = similarityTransform;

    return;
  }
}

//------------------------------------------------------------------------------
// This helper function completely copies the advanced combination transform
// elastix Transforms:
// AdvancedAffineTransformType, AdvancedTranslationTransformType
// AdvancedBSplineTransformType, AdvancedEulerTransformType,
// AdvancedSimilarityTransformType, AdvancedCombinationTransformType
template< class AdvancedAffineTransformType, class AdvancedTranslationTransformType,
          class AdvancedBSplineTransformType, class AdvancedEulerTransformType,
          class AdvancedSimilarityTransformType, class AdvancedCombinationTransformType >
void CopyAdvancedCombinationTransform(
  typename AdvancedCombinationTransformType::Pointer & advancedTransform,
  const typename AdvancedCombinationTransformType::CurrentTransformConstPointer & current )
{
  if ( current.IsNotNull() )
  {
    // Try Advanced Affine
    const typename AdvancedAffineTransformType::ConstPointer affine =
      dynamic_cast< const AdvancedAffineTransformType * >( current.GetPointer() );

    if ( affine )
    {
      // Create Advanced Affine transform
      typename AdvancedAffineTransformType::Pointer affineTransform =
        AdvancedAffineTransformType::New();

      affineTransform->SetFixedParameters( affine->GetFixedParameters() );
      affineTransform->SetParameters( affine->GetParameters() );

      advancedTransform->SetCurrentTransform( affineTransform );

      return;
    }

    // Try Advanced Translation
    const typename AdvancedTranslationTransformType::ConstPointer translation =
      dynamic_cast< const AdvancedTranslationTransformType * >( current.GetPointer() );

    if ( translation )
    {
      // Create Advanced Translation transform
      typename AdvancedTranslationTransformType::Pointer translationTransform =
        AdvancedTranslationTransformType::New();

      translationTransform->SetFixedParameters( translation->GetFixedParameters() );
      translationTransform->SetParameters( translation->GetParameters() );

      advancedTransform->SetCurrentTransform( translationTransform );

      return;
    }

    // Try Advanced BSpline
    const typename AdvancedBSplineTransformType::ConstPointer bspline =
      dynamic_cast< const AdvancedBSplineTransformType * >( current.GetPointer() );

    if ( bspline )
    {
      // Create Advanced BSpline transform
      typename AdvancedBSplineTransformType::Pointer bsplineTransform =
        AdvancedBSplineTransformType::New();

      // Set the same properties and grid
      bsplineTransform->SetGridOrigin( bspline->GetGridOrigin() );
      bsplineTransform->SetGridDirection( bspline->GetGridDirection() );
      bsplineTransform->SetGridSpacing( bspline->GetGridSpacing() );
      bsplineTransform->SetGridRegion( bspline->GetGridRegion() );

      bsplineTransform->SetParameters( bspline->GetParameters() );

      advancedTransform->SetCurrentTransform( bsplineTransform );

      return;
    }

    // Try Advanced Euler
    const typename AdvancedEulerTransformType::ConstPointer euler =
      dynamic_cast< const AdvancedEulerTransformType * >( current.GetPointer() );

    if ( euler )
    {
      // Create Advanced Euler transform
      typename AdvancedEulerTransformType::Pointer eulerTransform =
        AdvancedEulerTransformType::New();

      //eulerTransform->SetCenter( euler->GetCenter() );
      eulerTransform->SetFixedParameters( euler->GetFixedParameters() );
      eulerTransform->SetParameters( euler->GetParameters() );

      advancedTransform->SetCurrentTransform( eulerTransform );

      return;
    }

    // Try Advanced Similarity
    const typename AdvancedSimilarityTransformType::ConstPointer similarity =
      dynamic_cast< const AdvancedSimilarityTransformType * >( current.GetPointer() );

    if ( similarity )
    {
      // Create Advanced Similarity transform
      typename AdvancedSimilarityTransformType::Pointer similarityTransform =
        AdvancedSimilarityTransformType::New();

      //similarityTransform->SetCenter( similarity->GetCenter() );
      similarityTransform->SetFixedParameters( similarity->GetFixedParameters() );
      similarityTransform->SetParameters( similarity->GetParameters() );

      advancedTransform->SetCurrentTransform( similarityTransform );

      return;
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
int main( int argc, char *argv[] )
{
  // Check for GPU
  if ( !itk::IsGPUAvailable() )
  {
    std::cerr << "ERROR: OpenCL-enabled GPU is not present." << std::endl;
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

  if ( validateArguments == itk::CommandLineArgumentParser::FAILED )
  {
    return EXIT_FAILURE;
  }
  else if ( validateArguments == itk::CommandLineArgumentParser::HELPREQUESTED )
  {
    return EXIT_SUCCESS;
  }

  // Get command line arguments
  std::string inputFileName = "";
  const bool  retin = parser->GetCommandLineArgument( "-in", inputFileName );

  std::vector< std::string > outputFileNames( 2, "" );
  parser->GetCommandLineArgument( "-out", outputFileNames );

  // Get acceptable rmse error
  double rmseError;
  parser->GetCommandLineArgument( "-rmse", rmseError );

  // interpolator argument
  std::string interpolator = "NearestNeighbor";
  parser->GetCommandLineArgument( "-i", interpolator );

  if ( interpolator != "NearestNeighbor"
       && interpolator != "Linear"
       && interpolator != "BSpline" )
  {
    std::cerr << "ERROR: interpolator \"-i\" should be one of {NearestNeighbor, Linear, BSpline}." << std::endl;
    return EXIT_FAILURE;
  }

  // transform argument
  const bool                 useComboTransform = parser->ArgumentExists( "-c" );
  std::vector< std::string > transforms;
  transforms.push_back( "Affine" );
  parser->GetCommandLineArgument( "-t", transforms );

  // check that use combo transform provided when used multiple transforms
  if ( transforms.size() > 1 && !useComboTransform )
  {
    std::cerr << "ERROR: for multiple transforms option \"-c\" should provided." << std::endl;
    return EXIT_FAILURE;
  }

  // check for supported transforms
  for ( std::size_t i = 0; i < transforms.size(); i++ )
  {
    const std::string transformName = transforms[i];
    if ( transformName != "Affine"
         && transformName != "Translation"
         && transformName != "BSpline"
         && transformName != "Euler"
         && transformName != "Similarity" )
    {
      std::cerr << "ERROR: transforms \"-t\" should be one of "
                << "{Affine, Translation, BSpline, Euler, Similarity}"
                << " or combination of them." << std::endl;
      return EXIT_FAILURE;
    }
  }

  unsigned int runTimes = 1;
  std::string  parametersFileName = "";
  for ( std::size_t i = 0; i < transforms.size(); i++ )
  {
    if ( transforms[i] == "BSpline" )
    {
      const bool retp = parser->GetCommandLineArgument( "-p", parametersFileName );
      if ( !retp )
      {
        std::cerr << "ERROR: You should specify parameters file \"-p\" for the B-spline transform." << std::endl;
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
  elastix::SetupForDebugging();

  const unsigned int splineOrderInterpolator = 3;
  std::cout << std::showpoint << std::setprecision( 4 );

  // Typedefs.
  const unsigned int Dimension = 3;
  typedef short                                    InputPixelType;
  typedef short                                    OutputPixelType;
  typedef itk::Image< InputPixelType, Dimension >  InputImageType;
  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;
  typedef InputImageType::SizeType::SizeValueType  SizeValueType;

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

  // Interpolate typedefs
  typedef itk::InterpolateImageFunction<
      InputImageType, InterpolatorPrecisionType >             InterpolatorType;
  typedef itk::NearestNeighborInterpolateImageFunction<
      InputImageType, InterpolatorPrecisionType >             NearestNeighborInterpolatorType;
  typedef itk::LinearInterpolateImageFunction<
      InputImageType, InterpolatorPrecisionType >             LinearInterpolatorType;
  typedef itk::BSplineInterpolateImageFunction<
      InputImageType, ScalarType, InterpolatorPrecisionType > BSplineInterpolatorType;

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
  catch ( itk::ExceptionObject & e )
  {
    std::cerr << "ERROR: Caught ITK exception during cpuReader->Update(): " << e << std::endl;
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
  for ( std::size_t i = 0; i < Dimension; i++ )
  {
    tmp1 = randomNum->GetUniformVariate( 0.9, 1.1 );
    tmp2 = inputSpacing[i] * tmp1;
    s << tmp2; s >> outputSpacing[i]; s.clear();

    tmp1 = randomNum->GetUniformVariate( -10.0, 10.0 );
    tmp2 = inputOrigin[i] + tmp1;
    s << tmp2; s >> outputOrigin[i]; s.clear();

    for ( unsigned int j = 0; j < Dimension; j++ )
    {
      //tmp = randomNum->GetUniformVariate( 0.9 * inputOrigin[ i ], 1.1 *
      // inputOrigin[ i ] );
      outputDirection[i][j] = inputDirection[i][j];        // * tmp;
    }

    tmp1 = randomNum->GetUniformVariate( 0.9, 1.1 );
    outputSize[i] = itk::Math::Round< SizeValueType >( inputSize[i] * tmp1 );
  }

  cpuFilter->SetDefaultPixelValue( -1.0 );
  cpuFilter->SetOutputSpacing( outputSpacing );
  cpuFilter->SetOutputOrigin( outputOrigin );
  cpuFilter->SetOutputDirection( outputDirection );
  cpuFilter->SetSize( outputSize );
  cpuFilter->SetOutputStartIndex( inputRegion.GetIndex() );

  // Construct, select and setup transform
  if ( !useComboTransform )
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
      ( 0, transforms[0], cpuTransform, dummy, inputImage, bsplineParameters, parametersFileName );
  }
  else
  {
    AdvancedTransformType::Pointer            currentTransform;
    AdvancedCombinationTransformType::Pointer initialTransform;
    AdvancedCombinationTransformType::Pointer tmpTransform =
      AdvancedCombinationTransformType::New();
    initialTransform = tmpTransform;
    cpuTransform = tmpTransform;

    for ( std::size_t i = 0; i < transforms.size(); i++ )
    {
      if ( i == 0 )
      {
        SetTransform<
          // ITK Transforms
          TransformType, AffineTransformType, TranslationTransformType,
          BSplineTransformType, EulerTransformType, SimilarityTransformType,
          // elastix Transforms
          AdvancedCombinationTransformType, AdvancedAffineTransformType, AdvancedTranslationTransformType,
          AdvancedBSplineTransformType, AdvancedEulerTransformType, AdvancedSimilarityTransformType,
          InputImageType >
          ( i, transforms[i], cpuTransform, initialTransform, inputImage, bsplineParameters, parametersFileName );
      }
      else
      {
        AdvancedCombinationTransformType::Pointer initialNext =
          AdvancedCombinationTransformType::New();

        SetTransform<
          // ITK Transforms
          TransformType, AffineTransformType, TranslationTransformType,
          BSplineTransformType, EulerTransformType, SimilarityTransformType,
          // elastix Transforms
          AdvancedCombinationTransformType, AdvancedAffineTransformType, AdvancedTranslationTransformType,
          AdvancedBSplineTransformType, AdvancedEulerTransformType, AdvancedSimilarityTransformType,
          InputImageType >
          ( i, transforms[i], cpuTransform, initialNext, inputImage, bsplineParameters, parametersFileName );

        initialTransform->SetInitialTransform( initialNext );
        initialTransform = initialNext;
      }
    }
  }

  // Create CPU interpolator here
  DefineInterpolator< InterpolatorType, ScalarType >(
    cpuInterpolator, interpolator, splineOrderInterpolator );

  // Print info
  std::cout << "Testing the ResampleImageFilter, CPU vs GPU:\n";
  std::cout << "CPU/GPU transform interpolator #threads time speedup RMSE\n";

  // Time the filter, run on the CPU
  itk::TimeProbe cputimer;
  cputimer.Start();

  for ( std::size_t i = 0; i < runTimes; i++ )
  {
    cpuFilter->SetInput( cpuReader->GetOutput() );
    cpuFilter->SetTransform( cpuTransform );
    cpuFilter->SetInterpolator( cpuInterpolator );
    try
    {
      cpuFilter->Update();
    }
    catch ( itk::ExceptionObject & e )
    {
      std::cerr << "ERROR: Caught ITK exception during cpuFilter->Update(): " << e << std::endl;
      return EXIT_FAILURE;
    }

    // Modify the filter, only not the last iteration
    if ( i != runTimes - 1 )
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
  cpuWriter->SetFileName( outputFileNames[0].c_str() );
  try
  {
    cpuWriter->Update();
  }
  catch ( itk::ExceptionObject & e )
  {
    std::cerr << "ERROR: Caught ITK exception during cpuWriter->Update(): " << e << std::endl;
    return EXIT_FAILURE;
  }

  // Register object factory for GPU image and filter
  // All these filters that are constructed after this point are
  // turned into a GPU filter.
  itk::ObjectFactoryBase::RegisterFactory( itk::GPUImageFactory::New() );
  itk::ObjectFactoryBase::RegisterFactory( itk::GPUResampleImageFilterFactory::New() );
  // requires double support GPU
  itk::ObjectFactoryBase::RegisterFactory( itk::GPUCastImageFilterFactory::New() );

  // Transforms factory registration
  itk::ObjectFactoryBase::RegisterFactory( itk::GPUAffineTransformFactory::New() );
  itk::ObjectFactoryBase::RegisterFactory( itk::GPUTranslationTransformFactory::New() );
  itk::ObjectFactoryBase::RegisterFactory( itk::GPUBSplineTransformFactory::New() );
  itk::ObjectFactoryBase::RegisterFactory( itk::GPUEuler3DTransformFactory::New() );
  itk::ObjectFactoryBase::RegisterFactory( itk::GPUSimilarity3DTransformFactory::New() );

  // Interpolators factory registration
  itk::ObjectFactoryBase::RegisterFactory( itk::GPUNearestNeighborInterpolateImageFunctionFactory::New() );
  itk::ObjectFactoryBase::RegisterFactory( itk::GPULinearInterpolateImageFunctionFactory::New() );
  itk::ObjectFactoryBase::RegisterFactory( itk::GPUBSplineInterpolateImageFunctionFactory::New() );
  itk::ObjectFactoryBase::RegisterFactory( itk::GPUBSplineDecompositionImageFilterFactory::New() );

  // Advanced transforms factory registration
  itk::ObjectFactoryBase::RegisterFactory( itk::GPUAdvancedCombinationTransformFactory::New() );
  itk::ObjectFactoryBase::RegisterFactory( itk::GPUAdvancedMatrixOffsetTransformBaseFactory::New() );
  itk::ObjectFactoryBase::RegisterFactory( itk::GPUAdvancedTranslationTransformFactory::New() );
  itk::ObjectFactoryBase::RegisterFactory( itk::GPUAdvancedBSplineDeformableTransformFactory::New() );
  //itk::ObjectFactoryBase::RegisterFactory(
  // itk::GPUAdvancedEuler3DTransformFactory::New() );
  itk::ObjectFactoryBase::RegisterFactory( itk::GPUAdvancedSimilarity3DTransformFactory::New() );

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
    elastix::ITKObjectEnableWarnings( gpuFilter.GetPointer() );
  }
  catch ( itk::ExceptionObject & e )
  {
    std::cerr << "Caught ITK exception during gpuFilter::New(): " << e << std::endl;
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
  catch ( itk::ExceptionObject & e )
  {
    std::cerr << "ERROR: Caught ITK exception during cpuReader->Update(): " << e << std::endl;
    return EXIT_FAILURE;
  }

  try
  {
    if ( !useComboTransform )
    {
      CopyTransform< TransformType, AffineTransformType, TranslationTransformType,
        BSplineTransformType, EulerTransformType, SimilarityTransformType >
        ( cpuTransform, gpuTransform, bsplineParameters );
    }
    else
    {
      // Get CPU AdvancedCombinationTransform
      const AdvancedCombinationTransformType *CPUAdvancedCombinationTransform =
        dynamic_cast< const AdvancedCombinationTransformType * >( cpuTransform.GetPointer() );
      if ( CPUAdvancedCombinationTransform )
      {
        AdvancedTransformType::Pointer            currentTransform;
        AdvancedCombinationTransformType::Pointer initialTransform;
        AdvancedCombinationTransformType::Pointer tmpTransform =
          AdvancedCombinationTransformType::New();
        initialTransform = tmpTransform;
        gpuTransform = tmpTransform;

        for ( std::size_t i = 0; i < transforms.size(); i++ )
        {
          if ( i == 0 )
          {
            AdvancedCombinationTransformType::CurrentTransformConstPointer currentTransformCPU =
              CPUAdvancedCombinationTransform->GetCurrentTransform();

            CopyAdvancedCombinationTransform< AdvancedAffineTransformType, AdvancedTranslationTransformType,
              AdvancedBSplineTransformType, AdvancedEulerTransformType,
              AdvancedSimilarityTransformType, AdvancedCombinationTransformType >
              ( initialTransform, currentTransformCPU );
          }
          else
          {
            AdvancedCombinationTransformType::Pointer initialNext =
              AdvancedCombinationTransformType::New();

            AdvancedCombinationTransformType::InitialTransformConstPointer initialTransformCPU =
              CPUAdvancedCombinationTransform->GetInitialTransform();

            const AdvancedCombinationTransformType *initialTransformCPUCasted =
              dynamic_cast< const AdvancedCombinationTransformType * >( initialTransformCPU.GetPointer() );

            AdvancedCombinationTransformType::CurrentTransformConstPointer currentTransformCPU =
              initialTransformCPUCasted->GetCurrentTransform();

            CopyAdvancedCombinationTransform< AdvancedAffineTransformType, AdvancedTranslationTransformType,
              AdvancedBSplineTransformType, AdvancedEulerTransformType,
              AdvancedSimilarityTransformType, AdvancedCombinationTransformType >
              ( initialNext, currentTransformCPU );

            initialTransform->SetInitialTransform( initialNext );
            initialTransform = initialNext;
          }
        }
      }
      else
      {
        std::cerr << "ERROR: Unable to retrieve CPU AdvancedCombinationTransform." << std::endl;
        return EXIT_FAILURE;
      }
    }
  }
  catch ( itk::ExceptionObject & e )
  {
    std::cerr << "ERROR: Caught ITK exception during copy transforms: " << e << std::endl;
    return EXIT_FAILURE;
  }

  // Create GPU interpolator here
  DefineInterpolator< InterpolatorType, ScalarType >(
    gpuInterpolator, interpolator, splineOrderInterpolator );

  // Time the filter, run on the GPU
  itk::TimeProbe gputimer;
  gputimer.Start();
  for ( std::size_t i = 0; i < runTimes; i++ )
  {
    try
    {
      gpuFilter->SetInput( gpuReader->GetOutput() );
      gpuFilter->SetTransform( gpuTransform );
      gpuFilter->SetInterpolator( gpuInterpolator );
    }
    catch ( itk::ExceptionObject & e )
    {
      std::cerr << "ERROR: " << e << std::endl;
      return EXIT_FAILURE;
    }

    try
    {
      gpuFilter->Update();
    }
    catch ( itk::ExceptionObject & e )
    {
      std::cerr << "ERROR: " << e << std::endl;
      return EXIT_FAILURE;
    }
    // Due to some bug in the ITK synchronization we now manually
    // copy the result from GPU to CPU, without calling Update() again,
    // and not clearing GPU memory afterwards.
    itk::GPUExplicitSync< FilterType, OutputImageType >( gpuFilter, false, false );
    //itk::GPUExplicitSync<FilterType, ImageType>( gpuFilter, false, true ); //
    // crashes!

    // Modify the filter, only not the last iteration
    if ( i != runTimes - 1 )
    {
      gpuFilter->Modified();
    }
  }
  // GPU buffer has not been copied yet, so we have to make manual update
  //itk::GPUExplicitSync< FilterType, OutputImageType >( gpuFilter, false, false );
  gputimer.Stop();

  std::cout << "GPU " << cpuTransform->GetNameOfClass()
            << " " << cpuInterpolator->GetNameOfClass()
            << " x " << gputimer.GetMean() / runTimes
            << " " << cputimer.GetMean() / gputimer.GetMean();

  /** Write the GPU result. */
  WriterType::Pointer gpuWriter = WriterType::New();
  gpuWriter->SetInput( gpuFilter->GetOutput() );
  gpuWriter->SetFileName( outputFileNames[1].c_str() );
  try
  {
    gpuWriter->Update();
  }
  catch ( itk::ExceptionObject & e )
  {
    std::cerr << "ERROR: " << e << std::endl;
    return EXIT_FAILURE;
  }

  // Compute RMSE
  const double rmse = elastix::ComputeRMSE< double, OutputImageType, OutputImageType >
                        ( cpuFilter->GetOutput(), gpuFilter->GetOutput() );
  std::cout << " " << rmse << std::endl;

  // Check
  if ( rmse > rmseError )
  {
    std::cerr << "ERROR: the RMSE between the CPU and GPU results is "
              << rmse << ", which is larger than the expected "
              << rmseError << std::endl;
    return EXIT_FAILURE;
  }

  // End program.
  return EXIT_SUCCESS;
}
